from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError, RequestError, TransportError
import json
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

OUTPUT_FILE = "catalog_docs.json"

def get_es_credentials(env: str):
    env = env.upper()  # normalize (qa -> QA)
    host = os.getenv(f"{env}_ELASTIC_HOST")
    username = os.getenv(f"{env}_ELASTIC_USERNAME")
    password = os.getenv(f"{env}_ELASTIC_PASSWORD")

    if not all([host, username, password]):
        raise ValueError(f"Missing environment variables for {env}")
    return host, username, password

# ============================
# 🔌 Connect to Elasticsearch
# ============================

def connect_elasticsearch(host: str, user: str, pwd: str)-> Elasticsearch:
    """ 
    Create and return an Elasticsearch client.
    Change to cloud_id and API KEY once available #TO DO
    """ 
    
    es = Elasticsearch(
        host,
        basic_auth=(user, pwd),
        request_timeout=600,
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

     
def fetch_all_ecodes(es: Elasticsearch, index_name: str, output_file: str):
    """Fetch all documents (ecodes, name, attributes) from Elasticsearch and write to disk."""
    try:
        query_body = {
            "query": {
                "term": {
                "type.keyword": "style"
                }
            },
            "_source": ["_id", "name", "stringFacets", "searchAttributes"]
       }  
    # Initialize the scroll
        page = es.search(
            index=index_name,
            body=query_body,
            scroll="2m",      # Keep the context alive for 2 minutes
            size=1000         # Batch size per scroll
        )

        scroll_id = page["_scroll_id"]
        hits = page["hits"]["hits"]

        all_docs = []

        while hits:
            for hit in hits:
                all_docs.append({
                    "id": hit.get("_id"),
                    "name": hit["_source"].get("name", ""),
                    "searchAttributes": hit["_source"].get("searchAttributes", ""),
                    "stringFacets": hit["_source"].get("stringFacets", [])
                })

            # Fetch next batch
            page = es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = page["_scroll_id"]
            hits = page["hits"]["hits"]

        # Clear scroll context to free resources
        es.clear_scroll(scroll_id=scroll_id)

        save_to_file(all_docs,OUTPUT_FILE)
        print(f"✅ Successfully written {len(all_docs)} records to {output_file}")

    except Exception as e:
        print(f"❌ Error occurred: {e}")

def save_to_file(docs, file_path=OUTPUT_FILE):
    """Save documents to a JSON file with safe encoding."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(docs)} docs to {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"❌ Failed to write file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElasticSearch Config Loader")
    parser.add_argument("--env", type=str, required=True, help="Environment (DEV, QA, PRD)")
    args = parser.parse_args()
  
    try:
        host, user, pwd = get_es_credentials(args.env)
        print(f"✅ Connecting to {args.env.upper()} - {host} with user {user}")
        es = connect_elasticsearch(host, user, pwd) 
        catalog_docs = fetch_all_ecodes(es,'catalog-load-read', OUTPUT_FILE)
    except Exception as e:
        print(f"❌ Error: {e}")
