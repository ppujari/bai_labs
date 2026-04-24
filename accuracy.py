import pandas as pd
import json
import logging
from elasticsearch import Elasticsearch
import numpy as np
from sklearn.metrics import ndcg_score

# ------------------------------------------------------------
# Configuration placeholders (replace with your actual values)
# ------------------------------------------------------------
ES_HOST = "https://dsg-search-dev-east.es.eastus.azure.elastic-cloud.com"
ES_USER = "data_ingestion"
ES_PASS = "FVG8%7SMUys$dp4m"

#MODEL_ID = "intfloat__e5-small"
MODEL_ID = "sentence-transformers__all-minilm-l6-v2"
#MODEL_NAME = "e5"
MODEL_NAME = "sbert"
SEARCH_TYPE = "vector"
#INDEX_VECTOR = "pp-vs-e5-embeddings-v2"
INDEX_VECTOR = "pp-vs-sbert-embeddings-v1"
#INDEX_NAME = "pp-vs-e5-embeddings-v2"
INDEX_NAME = "pp-vs-sbert-embeddings-v1"
SIZE=32

# ------------------------------------------------------------
# Helper function to determine the embedding field name
# ------------------------------------------------------------
def get_embedding_column(model_name: str) -> str:
    return "vs_sbert_embedding" if model_name.lower() == "sbert" else "vs_e5_embedding"

# ------------------------------------------------------------
# Run KNN Query against Elasticsearch
# ------------------------------------------------------------
def run_query(es: Elasticsearch, query_text: str, size: int):
    """Run a KNN query against the Elasticsearch index."""
    search_field = get_embedding_column(MODEL_NAME)

    knn_query_body = {
        "size": size,
        "knn": {
            "field": search_field,
            "k": size,
            "num_candidates": size * 50,
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": MODEL_ID,
                    "model_text": query_text
                }
            }
        }
    }

    try:
        response = es.search(index=INDEX_VECTOR, body=knn_query_body)
        results = [
            {
                "score": hit["_score"],
                "title": hit["_source"].get("title", "N/A"),
                "id": hit.get("_id", "N/A")
            }
            for hit in response["hits"]["hits"]
        ]
        return results
    except Exception as e:
        logging.error(f"Query failed for text '{query_text}': {e}")
        return []

def precision_at_k(y_true, y_pred, k):
    """Precision@K"""
    if not y_pred:
        return 0.0
    if isinstance(y_pred, set):
        y_pred = list(y_pred)
    y_pred_k = y_pred[:k]
    return len(set(y_true) & set(y_pred_k)) / float(k or 1)


def recall_at_k(y_true, y_pred, k):
    """Recall@K"""
    if not y_pred:
        return 0.0
    if isinstance(y_pred, set):
        y_pred = list(y_pred)
    y_pred_k = y_pred[:k]
    return len(set(y_true) & set(y_pred_k)) / len(y_true) if y_true else 0.0

def reciprocal_rank(y_true, y_pred):
    """Reciprocal Rank (for one query)"""
    for i, p in enumerate(y_pred, start=1):
        if p in y_true:
            return 1.0 / i
    return 0.0

def ndcg_at_k(y_true, y_pred, k):
    """Compute NDCG@K using sklearn"""
    if not y_pred:
        return 0.0
    if isinstance(y_pred, set):
        y_pred = list(y_pred)
    
    # Binary relevance: 1 if relevant, 0 otherwise
    y_true_binary = [1 if p in y_true else 0 for p in y_pred[:k]]
    
    if not any(y_true_binary):
        return 0.0
    
    # Use actual relevance scores (binary in this case)
    # sklearn expects [true_relevance], [predicted_scores]
    predicted_scores = list(range(k, 0, -1))  # Higher rank = higher score
    return ndcg_score([y_true_binary], [predicted_scores])

def evaluate_metrics(results, ground_truth, k=16):
    """
    results: dict -> {query: [predicted_ecodes]}
    ground_truth: dict -> {query: [true_ecodes]}
    """
    metrics = []
    for query, preds in results.items():
        true = ground_truth.get(query, [])
        if not true:
            continue
        p = precision_at_k(true, preds, k)
        r = recall_at_k(true, preds, k)
        mrr = reciprocal_rank(true, preds)
        ndcg = ndcg_at_k(true, preds, k)
        metrics.append((query, p, r, mrr, ndcg))
    
    df = pd.DataFrame(metrics, columns=["query", f"P@{k}", f"R@{k}", "MRR", f"NDCG@{k}"])
    return df, df[[f"P@{k}", f"R@{k}", "MRR", f"NDCG@{k}"]].mean().to_dict()

# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main():
    input_excel = "~/Downloads/119 TEXT Vector Search Examples.xlsx"
    output_file = "matched_results.csv"

    # Step 1: Read Excel file
    try:
        df = pd.read_excel(input_excel)
        logging.info(f"Loaded {len(df)} rows from Excel file.")
    except Exception as e:
        raise RuntimeError(f"Error reading Excel file: {e}")

    # Step 2: Connect to Elasticsearch
    try:
        es = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USER, ES_PASS),
            request_timeout=60,
            max_retries=3,
            retry_on_timeout=True
        )
        if not es.ping():
            raise ConnectionError("Elasticsearch cluster not reachable.")
        logging.info("✅ Connected to Elasticsearch successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Elasticsearch: {e}")

    # Step 3: Process each search term
    results = []
    ecode_cols = df.columns[2:]  # skip first column (search term)
    ground_truth = {}
    results_sbert = {}

    for _, row in df.iterrows():
        search_term = str(row[df.columns[1]]).strip()
        if not search_term:
            continue

        # Collect expected ecodes (convert to uppercase)
        expected_ecodes = {
            str(row[col]).strip().upper()
            for col in ecode_cols
            if pd.notna(row[col]) and str(row[col]).strip()
        }
        # Run KNN query
        query_text = search_term
        retrieved = run_query(es, query_text, size=SIZE)

        # Extract retrieved ecodes
        retrieved_ecodes = {r["id"].upper() for r in retrieved}

        # Find matches
        matches = expected_ecodes.intersection(retrieved_ecodes)

        if matches:
            results.append({
                "search_term": search_term,
                "matched_ecodes": ", ".join(sorted(matches))
            })
        ground_truth[search_term]=expected_ecodes
        results_sbert[search_term]=retrieved_ecodes

    # Step 4: Save output to CSV
    if not results:
        logging.warning("No matches found for any queries!")
        output_df = pd.DataFrame(columns=["search_term", "matched_ecodes"])
    else:
        output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    logging.info(f"✅ Output saved to {output_file}")
    print(f"✅ Output saved to {output_file}")
    df_sbert, avg_sbert = evaluate_metrics(results_sbert, ground_truth, k=SIZE)

    print(f"SBERT average metrics: { {k: round(v, 2) for k, v in avg_sbert.items()} }")

# ------------------------------------------------------------
# Run main
# ------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

