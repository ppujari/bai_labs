#!/usr/bin/env python3

"""
This script is beginers guide to ELSER model search.
1. Connects to Elasticsearch

2. Creates an ingest pipeline using your inference endpoint

3. Creates an index (pp_vs_elser) with a sparse_vector field

4. Indexes documents from a pandas DataFrame through the pipeline

5. Runs a search query using the sparse_vector query
"""
import os
import json
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
from dotenv import load_dotenv
load_dotenv()

# ================================
# Constants
# ================================
ELSER_MODEL_ID = ".elser_model_2"                  # Model deployed in your cluster
PIPELINE_ID = "elser-search-pipeline"              # Ingest pipeline ID
INDEX_NAME = "pp-vs-embeddings-v1"                 # Index name
PIPELINE_ID = "elser-search-pipeline"
SHOULD_DELETE_INDEX = False

# Load raw file as text
def get_data(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract _id, title, searchAttributes
    cleaned = []
    for doc in data:
        #source = {k: v for k, v in doc["_source"].items() if k != "stringFacets"}
        source = {k: v for k, v in doc.items() if k != "stringFacets"}
        source["_id"] = doc["_id"]  # Add the _id from Elasticsearch
        cleaned.append(source)

    # Load into DataFrame
    df = pd.DataFrame(cleaned)
    #in case of missing values or NaN
    df['title_plus_attributes'] = (df['name'].fillna('') + " " + df['searchAttributes'].fillna('')).str.strip()

    return df

# ================================
# 0. Create connection to ES 
# ================================

def get_es_connection(env):
    ES_HOST = os.getenv(f"ELASTIC_HOST_{env}")
    ES_USER = os.getenv("ELASTIC_USERNAME")
    ES_PASS = os.getenv("ELASTIC_PASSWORD")
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER,ES_PASS),
        request_timeout=600,
        verify_certs=True  # Optional: Set to False only if using self-signed certs
    )

    # Test connection
    print(es.info().body)
    return es 

# ================================
# 1. Create ingest pipeline for  ELSER
# ================================

def ensure_pipeline(es, pipeline_id, pipeline_body):
    try:
        es.ingest.get_pipeline(id=pipeline_id)
        print(f"ℹ️ Pipeline '{pipeline_id}' already exists. Skipping creation.")
    except NotFoundError:
        es.ingest.put_pipeline(
            id=pipeline_id,
            processors=pipeline_body["processors"],
            description=pipeline_body["description"]
        )
        print(f"✅ Pipeline '{pipeline_id}' created.")


# ================================
# 2. Create index with sparse_vector mapping
# ================================
def create_index(es : Elasticsearch):
    """
    Create the index with a sparse_vector field to hold ELSER embeddings.
    """
    if SHOULD_DELETE_INDEX and es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    mapping = {
        "mappings": {
            "properties": {
                "title_plus_attributes": {"type": "text"},
                "content_embedding": {"type": "sparse_vector"}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"✅ Index '{INDEX_NAME}' created with sparse_vector field.")

# ================================
# 3. Index documents from DataFrame
# ================================
def index_dataframe(df):
    """
    Index documents from a pandas DataFrame into Elasticsearch.
    Requires the ingest pipeline to enrich with embeddings.
    """
    actions = []
    for _, row in df.iterrows():
        actions.append({
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": str(row['_id']),
            "pipeline": "elser-search-pipeline",
            "_source": {
                "title_plus_attributes": row['title_plus_attributes']
            }
        })

    success, failed = helpers.bulk(es, actions, stats_only=True, refresh="wait_for")

    print(f"✅ Indexed {success} documents.")
    if failed:
        print(f"⚠️ {failed} documents failed.")

# ================================
# 4. Semantic search using sparse_vector query
# ================================
def search_semantic(es:Elasticsearch, query_text, top_k=3):
    """
    Perform semantic search using the sparse_vector query with ELSER inference.
    """
    response = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {
                "sparse_vector": {
                    "field": "content_embedding",     # matches the target_field in pipeline
                    "query": query_text,
                    "inference_id":"mydsg-elser-endpoint"
                }
            }
        }
    )

    total_hits = response['hits']['total']['value']
    print(total_hits)
    hits = response["hits"]["hits"]
    for hit in hits:
        print(f"🔎 Score: {hit['_score']:.4f} | Content: {hit['_source']['title_plus_attributes']}\n")

# ================================
# Example usage
# ================================
if __name__ == "__main__":
    path = '/Users/dks0802651/work/vector_search/data/catalog_docs.json'
    df = get_data(path)
    es = get_es_connection("DEV")
    # Setup pipeline + index
    pipeline_body = {
        "description": "Ingest pipeline for ELSER model",
        "processors": [
            {
                "inference": {
                    "model_id":ELSER_MODEL_ID,
                    "input_output": {
                        "input_field": "title_plus_attributes",
                        "output_field": "content_embedding"
                    }
                }
            }
        ]
    }
    ensure_pipeline(es, PIPELINE_ID, pipeline_body)
    create_index(es)

    # Index data
    index_dataframe(df)
    # Query
    query_text = "comfortable running shoes"
    search_semantic(es,query_text)
