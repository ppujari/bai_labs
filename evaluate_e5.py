import pandas as pd
import logging
from elasticsearch import Elasticsearch
import numpy as np
from sklearn.metrics import ndcg_score
import os

# Configuration
ES_HOST = "host_name"
ES_USER = os.getenv("ES_USER", "user_name")
ES_PASS = os.getenv("ES_PASS", "pass")

# Models to compare
MODELS = {
    "E5": {
        "model_id": "intfloat__e5-small",
        "index": "pp-vs-e5-embeddings-v1",
        #"index": "pp-vs-e5-llmd-embeddings",
        "field": "vs_e5_embedding",
        #"field": "vs_e5_llm_product_desc_embedding",
        "prefix_query": True  # E5 needs "query:" prefix
    },
    "SBERT": {
        "model_id": "sentence-transformers__all-minilm-l6-v2",
        "index": "pp-vs-sbert-active-ecodes-embeddings",
        #"index": "pp-vs-sbert-llmd-embeddings",
        "field": "vs_sbert_embedding",
        #"field": "vs_sbert_llm_product_desc_embedding",
        "prefix_query": False
    }
}

SIZE = 32

def run_vector_search(es, model_config, query_text, size=32):
    """
    Pure vector search - NO keyword matching
    """
    model_id = model_config["model_id"]
    index_name = model_config["index"]
    search_field = model_config["field"]
    
    # Add "query:" prefix for E5
    processed_query = query_text
    if model_config.get("prefix_query"):
        processed_query = f"query: {query_text}"
    
    # Pure KNN query - vector only
    knn_query_body = {
        "size": size,
        "knn": {
            "field": search_field,
            "k": size,
            "num_candidates": size * 30,  # 30x for better accuracy
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": model_id,
                    "model_text": processed_query
                }
            }
        },
        "_source": ["title"]
    }

    try:
        response = es.search(index=index_name, body=knn_query_body)
        results = [hit.get("_id", "N/A").upper() for hit in response["hits"]["hits"]]
        return results
    except Exception as e:
        logging.error(f"Query failed for '{query_text}': {e}")
        return []

def precision_at_k(y_true, y_pred, k):
    """Precision@K"""
    if not y_pred:
        return 0.0
    y_pred_k = y_pred[:k]
    return len(set(y_true) & set(y_pred_k)) / float(k)

def recall_at_k(y_true, y_pred, k):
    """Recall@K"""
    if not y_pred:
        return 0.0
    y_pred_k = y_pred[:k]
    return len(set(y_true) & set(y_pred_k)) / len(y_true) if y_true else 0.0

def reciprocal_rank(y_true, y_pred):
    """Reciprocal Rank"""
    for i, p in enumerate(y_pred, start=1):
        if p in y_true:
            return 1.0 / i
    return 0.0

def ndcg_at_k(y_true, y_pred, k):
    """Compute NDCG@K"""
    if not y_pred:
        return 0.0
    
    y_true_binary = [1 if p in y_true else 0 for p in y_pred[:k]]
    if not any(y_true_binary):
        return 0.0
    
    predicted_scores = list(range(k, 0, -1))
    return ndcg_score([y_true_binary], [predicted_scores])

def evaluate_metrics(results, ground_truth, k):
    """Calculate metrics"""
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

def main():
    input_excel = os.path.expanduser("~/Downloads/119 TEXT Vector Search Examples.xlsx")
    
    # Load data
    try:
        df = pd.read_excel(input_excel)
        logging.info(f"Loaded {len(df)} rows from Excel file.")
    except Exception as e:
        raise RuntimeError(f"Error reading Excel file: {e}")

    # Connect to Elasticsearch
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

    # Prepare ground truth
    ecode_cols = df.columns[2:]
    ground_truth = {}
    
    for _, row in df.iterrows():
        search_term = str(row[df.columns[1]]).strip()
        if not search_term:
            continue
        
        expected_ecodes = [
            str(row[col]).strip().upper()
            for col in ecode_cols
            if pd.notna(row[col]) and str(row[col]).strip()
        ]
        ground_truth[search_term] = expected_ecodes

    # Compare models
    print("\n" + "="*80)
    print("VECTOR SEARCH COMPARISON (Pure Semantic Search)")
    print("="*80)
    
    all_results = {}
    
    for model_name, model_config in MODELS.items():
        print(f"\n🔍 Testing {model_name} model...")
        print(f"   Model: {model_config['model_id']}")
        print(f"   Index: {model_config['index']}")
        
        results = {}
        
        for search_term in ground_truth.keys():
            retrieved = run_vector_search(es, model_config, search_term, size=SIZE)
            results[search_term] = retrieved
        
        # Calculate metrics
        df_metrics, avg_metrics = evaluate_metrics(results, ground_truth, k=SIZE)
        
        # Save detailed results
        output_file = f"vector_comparison_{model_name.lower()}_detailed.csv"
        df_metrics.to_csv(output_file, index=False)
        
        all_results[model_name] = avg_metrics
        
        print(f"\n📊 {model_name} Results:")
        for metric, value in avg_metrics.items():
            print(f"   {metric}: {value:.3f}")

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame(all_results).T
    print(comparison_df.to_string())
    
    # Save comparison
    comparison_df.to_csv("vector_model_comparison.csv")
    print(f"\n✅ Comparison saved to: vector_model_comparison.csv")
    
    # Highlight winner
    print("\n" + "="*80)
    print("WINNER BY METRIC:")
    print("="*80)
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"   {metric}: {best_model} ({best_score:.3f})")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

