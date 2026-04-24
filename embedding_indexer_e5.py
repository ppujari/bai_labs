#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name:embedding_indexer.py
Description: creates and indexes embeddings for title and search attributes using E5 and Sentence BERT model
fields collected from ES index are as follows:
Data:
_id
name
searchAttributes
stringFacets

Author: Pradeep Pujari
Date: 2025-08-22
"""
import os
import json
import time
import certifi
import pandas as pd
import numpy as np
from typing import List, Dict, Generator, Any
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
from elasticsearch.exceptions import TransportError, NotFoundError, RequestError
from dotenv import load_dotenv
load_dotenv()
# =========================================================================================
# 🔧 Configuration Constants
# Option 2: Move them to a separate config.py file
# If your project is growing, it's better to isolate constants in a separate config module:
# ==========================================================================================
env = "DEV"
MODEL_ID = "intfloat__e5-small"
#MODEL_ID = ".multilingual-e5-small"
MODEL_NAME = "e5"
SEARCH_TYPE = "vector"
INDEX_RAW = "pp-vs-raw-docs-v1"       # stores original raw documents
INDEX_VECTOR = "pp-vs-e5-embeddings-v1"  # stores documents with vector embeddings
INDEX_NAME = "pp-vs-e5-embeddings-v1"
PIPELINE_ID = f"{MODEL_NAME.lower()}_embedding_pipeline"
SHOULD_DELETE_INDEX = False 
#INPUT_FIELD = "flat_description"
INPUT_FIELD = "title_plus_attributes"
INDEX_SETTINGS = {
    "index": {
        "number_of_replicas": "0",
        "number_of_shards": "1",
        #"knn": True  # Enable kNN search
        # No need for "knn": true — only needed if using legacy knn_vector
    }
}
#both sbert and e5 small has same dimension.
dims = 384 if MODEL_NAME.lower() == "sbert" else 384  # adjust per model

def load_data(path: str) -> pd.DataFrame:
    """Load cleaned JSON data into a pandas DataFrame"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten all rows in a DataFrame and add a flat_description column, 
    using identifiers directly from stringFacets without a hardcoded map.
    """
    df.dropna(subset=['name'], inplace=True)
    df = df.fillna(value=np.nan)  # ensure proper NaNs
    df = df.where(pd.notnull(df), None)  # replace NaN with None (JSON null)


    if "name" not in df.columns:
        df["name"] = ""
    if "searchAttributes" not in df.columns:
        df["searchAttributes"] = ""

    df["title_plus_attributes"] = (
        "passage: "
        + df["name"].fillna("").astype(str)
        + " "
        + df["searchAttributes"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df[df["title_plus_attributes"].str.len() > len("passage: ")]
    #df = df[df["title_plus_attributes"].str.len() > 1] 


    print(df.head())
    return df

def get_embedding_column(model_name: str) -> str:
    return "vs_sbert_embedding" if model_name.lower() == "sbert" else "vs_e5_embedding"

# ============================
# 🔌 Connect to Elasticsearch
# ============================

def connect_elasticsearch()-> Elasticsearch:
    """
    Create and return an Elasticsearch client.
    Change to cloud_id and API KEY once available #TO DO
    """
    ES_HOST = os.getenv(f"ELASTIC_HOST_{env}")
    ES_USER = os.getenv("ELASTIC_USERNAME")
    ES_PASS = os.getenv("ELASTIC_PASSWORD")
    
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        request_timeout=30,
        http_compress=False,
        ca_certs=certifi.where(),  # ensures trusted CA bundle
        verify_certs=True
    )

    try:
        info = es.info().body
        print("✅ Connected to Elasticsearch:")
        print(info)
    except Exception as e:
        print("❌ Failed to connect to Elasticsearch.")
        raise e

    return es


def define_ingest_pipeline(es: Elasticsearch, model_id: str, model_name: str) -> None:
    """Create an ingest pipeline that generates embeddings for E5/SBERT."""
    embedding_column = get_embedding_column(model_name)

    pipeline_body = {
        "description": "Elastic product info embedding pipeline with null checks",
        "processors": [
            {
                "inference": {
                    "model_id": model_id,
                    "input_output": {
                        "input_field": INPUT_FIELD,
                        "output_field": embedding_column
                    },
                    "ignore_missing": True,
                    "if": f"ctx.containsKey('{INPUT_FIELD}') && ctx['{INPUT_FIELD}'] != null && ctx['{INPUT_FIELD}'] instanceof String && ctx['{INPUT_FIELD}'].trim().length() > 0"
                }
            }
        ],
        "on_failure": [
            {
                "set": {
                    "field": "error",
                    "value": "{{ _ingest.on_failure_message }}"
                }
            }
        ]
    }

    try:
        es.ingest.put_pipeline(id=PIPELINE_ID, body=pipeline_body)
        print(f"✅ Ingest pipeline '{PIPELINE_ID}' created for model {model_id}, output field: {embedding_column}")
    except TransportError as e:
        print(f"❌ Failed to create ingest pipeline '{PIPELINE_ID}': {e}")


def get_index_mapping(model_name: str = "sbert", include_vector: bool = True) -> Dict[str, Any]:
    """Generate index mapping for text-only (raw) or text+vector (vector index)."""

    # Common text fields
    text_fields: Dict[str, Any] = {
        "title": {
            "type": "text",
            "fields": {
                "keyword": {"type": "keyword", "ignore_above": 256}
            }
        },
        INPUT_FIELD: {"type": "text"},
        "ecode": {"type": "keyword"},
    }

    # Build base mapping
    properties = {**text_fields}

    if include_vector:
        vector_field_name = get_embedding_column(model_name)
        properties[vector_field_name] = {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        }

    return {
        "mappings": {
            "properties": properties
        }
    }

def bulk_index_raw(es: Elasticsearch, index_name: str, df: pd.DataFrame) -> None:
    """Stage 1: Index raw documents into ES (no embeddings)."""
    actions = []
    for _, row in df.iterrows():
        doc = {
            "_index": index_name,
            "_id": row["id"],
            "_source": {
                "title": row["name"],
                INPUT_FIELD: row.get(INPUT_FIELD, "")
            }
        }
        actions.append(doc)
    try:
        helpers.bulk(es, actions, index=index_name)
        print(f"✅ Successfully indexed {len(actions)} documents into {index_name}")
    except BulkIndexError as e:
        print("❌ Bulk indexing failed for some/all documents")
        for error in e.errors[:5]:  # show first 5
            print(error)
    except TransportError as e:
        print(f"❌ Transport-level error during bulk indexing: {e}")
    
    try:
        es.indices.refresh(index=index_name)
        print(f"✅ Index {index_name} refreshed")
    except TransportError as e:
        print(f"⚠️ Failed to refresh index {index_name}: {e}")


def reindex_with_pipeline(es: Elasticsearch) -> None:
    """Stage 2: Reindex raw docs through pipeline to generate embeddings (TLS-safe async version)."""
    body = {
        "source": {"index": INDEX_RAW},
        "dest": {"index": INDEX_VECTOR, "pipeline": PIPELINE_ID}
    }

    try:
        # Start reindex asynchronously
        resp = (es.options(request_timeout=600)
          .reindex(body=body, wait_for_completion=False, refresh=True))
        task_id = resp.get("task")
        if not task_id:
            print("❌ Reindex did not return a valid task ID.")
            return
        print(f"✅ Reindex started asynchronously. Task ID: {task_id}")

        # Poll for completion
        while True:
            try:
                status = es.tasks.get(task_id=task_id, request_timeout=30)

                # Completed?
                if status.get("completed"):
                    response = status.get("response", {})
                    created = response.get("created") or response.get("total", 0)
                    updated = response.get("updated", 0)
                    failures = response.get("failures", [])
                    print(f"✅ Reindex completed. Created: {created}, Updated: {updated}, Failures: {len(failures)}")
                    break

                # Print progress stats if available
                progress = status.get("status", {})
                total = progress.get("total", 0)
                created = progress.get("created", 0)
                print(f"⏳ Progress: {created}/{total} documents reindexed...")
                time.sleep(60)  # poll every 60s

            except TransportError as e:
                print(f"⚠️ Polling error (retrying in 30s): {e}")
                time.sleep(30)
                continue

    except TransportError as e:
        print(f"❌ Failed to start reindex task: {e}")


def create_index(es: Elasticsearch, index_name: str, index_mapping: Dict[str, Any]) -> None:
    """Creates an empty index with the given name and mapping."""
    try:
        # Delete if required
        if SHOULD_DELETE_INDEX and es.indices.exists(index=index_name):
            print(f"Deleting existing {index_name}")
            es.options(ignore_status=[400, 404]).indices.delete(index=INDEX_NAME)

        # Create index if it doesn't exist
        if not es.indices.exists(index=index_name):
            body = {
                "settings": INDEX_SETTINGS,
                "mappings": {
                    "properties": index_mapping["mappings"]["properties"]
                }
            }
            es.indices.create(index=index_name, body=body)
            print(f"✅ Index '{index_name}' created successfully.")
        else:
            print(f"ℹ️ Index '{index_name}' already exists, skipping creation.")

    except TransportError as e:
        print(f"❌ Failed to create index '{index_name}': {e}")

def run_query(es: Elasticsearch, query_text: str, size: int = 5):
    """Run a KNN query against the index"""
    search_field = get_embedding_column(MODEL_NAME)
    
    knn_query_body = {
        "knn": {
            "field": search_field,
            "k": size,  # Number of nearest neighbors to retrieve
            "num_candidates": size * 10,  # Number of candidates to consider during search
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": MODEL_ID,    # deployed SBERT/E5 model
                    "model_text": query_text      # the query text to embed
                }
            }
        }
    }
    try:
        response = es.search(index=INDEX_VECTOR,body=knn_query_body)
        hits = response["hits"]["hits"]
        print(f"✅ Retrieved {len(hits)} results")
    except NotFoundError:
        print(f"❌ Index {INDEX_VECTOR} not found")
    except RequestError as e:
        print(f"❌ Bad query: {e.info}")
    except TransportError as e:
        print(f"⚠️ Transport error: {e}")

    results = [
        {
            "score": hit["_score"],
            "title": hit["_source"].get("title", "N/A"),
            "id": hit.get("_id", "N/A")
        }
        for hit in response["hits"]["hits"]
    ]

    print(json.dumps(results, indent=2))

# ======================
# 🧪 Entry Point
#Raw index → vector index → pipeline → bulk index raw → reindex → query
# ======================
def main()-> None:
    path = "./data/catalog_active_ecodes.json"
    df = load_data(path)
    df = prepare_data(df)

    es = connect_elasticsearch()

    # Prepare index mappings
    raw_index_mapping = get_index_mapping(model_name=MODEL_NAME, include_vector=False)
    vector_index_mapping = get_index_mapping(model_name=MODEL_NAME, include_vector=True)

    # Create indexes
    create_index(es, INDEX_RAW, raw_index_mapping)
    create_index(es, INDEX_VECTOR, vector_index_mapping)

    # Create the pipeline (used in reindexing)
    define_ingest_pipeline(es, MODEL_ID, MODEL_NAME)

    # Step 1: Bulk index raw docs (text only)
    bulk_index_raw(es, INDEX_RAW, df)

    # Step 2: Reindex with pipeline to generate embeddings
    reindex_with_pipeline(es)
    
    # Test query
    query_text = f"query: comfortable nike running shoes for men"
    run_query(es, query_text)

if __name__ == "__main__":
    main()
