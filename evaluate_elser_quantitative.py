import pandas as pd
import logging
from elasticsearch import Elasticsearch
import numpy as np
from sklearn.metrics import ndcg_score
import os

# Configuration
ES_HOST = "https://dsg-search-dev-east.es.eastus.azure.elastic-cloud.com"
ES_USER = os.getenv("ES_USER", "data_ingestion")
ES_PASS = os.getenv("ES_PASS", "FVG8%7SMUys$dp4m")

# Model configurations
MODELS = {
    "E5": {
        "model_id": "intfloat__e5-small",
        "index": "pp-vs-e5-llmd-embeddings",
        "field": "vs_e5_llm_product_desc_embedding",
        "type": "dense",
        "prefix_query": True
    },
    "ELSER": {
        "model_id": ".elser_model_2",
        "index": "pp-vs-elser-llmd-embeddings",
        "field": "content_embedding",
        "type": "sparse",
        "prefix_query": False
    }
}

SIZE = 32

def run_vector_search(es, model_name, model_config, query_text, size=32):
    """
    Run search for either E5 (dense) or ELSER (sparse)
    """
    index_name = model_config["index"]
    
    try:
        if model_config["type"] == "dense":
            # E5 dense vector search
            model_id = model_config["model_id"]
            search_field = model_config["field"]
            
            # Add "query:" prefix for E5
            processed_query = f"query: {query_text}" if model_config.get("prefix_query") else query_text
            
            query_body = {
                "size": size,
                "knn": {
                    "field": search_field,
                    "k": size,
                    "num_candidates": size * 30,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": model_id,
                            "model_text": processed_query
                        }
                    }
                },
                "_source": ["title"]
            }
            
        else:  # ELSER sparse vector search
            query_body = {
                "size": size,
                "query": {
                    "text_expansion": {
                        model_config["field"]: {
                            "model_id": model_config["model_id"],
                            "model_text": query_text
                        }
                    }
                },
                "_source": ["title"]
            }
        
        response = es.search(index=index_name, body=query_body)
        results = [hit.get("_id", "N/A").upper() for hit in response["hits"]["hits"]]
        return results
        
    except Exception as e:
        logging.error(f"Query failed for {model_name} '{query_text}': {e}")
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

def evaluate_model(es, model_name, model_config, ground_truth, k=32):
    """
    Evaluate a single model
    """
    results = {}
    total_queries = len(ground_truth)
    
    print(f"\n   Processing {total_queries} queries...")
    
    for idx, search_term in enumerate(ground_truth.keys(), 1):
        if idx % 10 == 0:
            print(f"   Progress: {idx}/{total_queries} queries processed")
        
        retrieved = run_vector_search(es, model_name, model_config, search_term, size=k)
        results[search_term] = retrieved
    
    # Calculate metrics
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
    avg_metrics = df[[f"P@{k}", f"R@{k}", "MRR", f"NDCG@{k}"]].mean().to_dict()
    
    return df, avg_metrics

def quantitative_comparison(es, ground_truth):
    """
    Run quantitative comparison only
    """
    print("\n" + "="*80)
    print("QUANTITATIVE EVALUATION: E5 vs ELSER")
    print("="*80)
    print(f"\nTest Dataset: {len(ground_truth)} search queries")
    print(f"Evaluation Metrics: P@{SIZE}, R@{SIZE}, MRR, NDCG@{SIZE}")
    print(f"Document Corpus: 200K products\n")
    
    all_metrics = {}
    all_dataframes = {}
    
    # Evaluate each model
    for model_name, model_config in MODELS.items():
        print("=" * 80)
        print(f"🔍 Evaluating {model_name} Model")
        print("=" * 80)
        print(f"   Model ID: {model_config['model_id']}")
        print(f"   Index: {model_config['index']}")
        print(f"   Type: {model_config['type'].upper()}")
        
        df, avg_metrics = evaluate_model(
            es, model_name, model_config, ground_truth, k=SIZE
        )
        
        all_metrics[model_name] = avg_metrics
        all_dataframes[model_name] = df
        
        # Display results for this model
        print(f"\n   📊 Results:")
        print(f"      P@{SIZE}:    {avg_metrics[f'P@{SIZE}']:.4f}")
        print(f"      R@{SIZE}:    {avg_metrics[f'R@{SIZE}']:.4f}")
        print(f"      MRR:      {avg_metrics['MRR']:.4f}")
        print(f"      NDCG@{SIZE}: {avg_metrics[f'NDCG@{SIZE}']:.4f}")
        
        # Save detailed results
        output_file = f"quantitative_{model_name.lower()}_detailed.csv"
        df.to_csv(output_file, index=False)
        print(f"\n   ✅ Detailed results saved to: {output_file}")
    
    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("📊 QUANTITATIVE COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_metrics).T
    print("\n" + comparison_df.to_string())
    
    # Save comparison summary
    comparison_df.to_csv("quantitative_comparison_summary.csv")
    print("\n✅ Summary saved to: quantitative_comparison_summary.csv")
    
    # ========================================================================
    # METRIC-BY-METRIC ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("🏆 WINNER BY METRIC")
    print("="*80)
    
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        worst_score = comparison_df[metric].min()
        
        if worst_score > 0:
            improvement = ((best_score / worst_score) - 1) * 100
            print(f"\n   {metric}:")
            print(f"      Winner: {best_model} ({best_score:.4f})")
            print(f"      Improvement: {improvement:+.1f}% better than other model")
        else:
            print(f"\n   {metric}:")
            print(f"      Winner: {best_model} ({best_score:.4f})")
    
    # ========================================================================
    # OVERALL ASSESSMENT
    # ========================================================================
    print("\n" + "="*80)
    print("💡 OVERALL ASSESSMENT")
    print("="*80)
    
    e5_ndcg = all_metrics['E5'][f'NDCG@{SIZE}']
    elser_ndcg = all_metrics['ELSER'][f'NDCG@{SIZE}']
    
    if elser_ndcg > e5_ndcg:
        improvement = ((elser_ndcg / e5_ndcg) - 1) * 100
        print(f"\n✅ ELSER outperforms E5 by {improvement:.1f}% (NDCG)")
        better_model = "ELSER"
    else:
        improvement = ((e5_ndcg / elser_ndcg) - 1) * 100
        print(f"\n✅ E5 outperforms ELSER by {improvement:.1f}% (NDCG)")
        better_model = "E5"
    
    # Check if performance is acceptable
    acceptable_threshold = 0.40
    max_ndcg = max(e5_ndcg, elser_ndcg)
    
    if max_ndcg < acceptable_threshold:
        print(f"\n⚠️  WARNING: Both models below acceptable threshold")
        print(f"   Current best NDCG: {max_ndcg:.3f}")
        print(f"   Target NDCG: {acceptable_threshold:.3f}")
        print(f"   Gap: {(acceptable_threshold - max_ndcg):.3f}")
    else:
        print(f"\n✅ Best model ({better_model}) meets acceptable threshold")
        print(f"   NDCG: {max_ndcg:.3f} (target: {acceptable_threshold:.3f})")
    
    # ========================================================================
    # STATISTICAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("📈 STATISTICAL SUMMARY")
    print("="*80)
    
    for model_name, df in all_dataframes.items():
        print(f"\n{model_name} Model Statistics:")
        print(f"   Min NDCG@{SIZE}:  {df[f'NDCG@{SIZE}'].min():.4f}")
        print(f"   Max NDCG@{SIZE}:  {df[f'NDCG@{SIZE}'].max():.4f}")
        print(f"   Median NDCG@{SIZE}: {df[f'NDCG@{SIZE}'].median():.4f}")
        print(f"   Std Dev:     {df[f'NDCG@{SIZE}'].std():.4f}")
        
        # Count queries with no results
        zero_ndcg = (df[f'NDCG@{SIZE}'] == 0).sum()
        print(f"   Zero NDCG queries: {zero_ndcg}/{len(df)} ({zero_ndcg/len(df)*100:.1f}%)")
    
    return all_metrics, comparison_df

def main():
    input_excel = os.path.expanduser("~/Downloads/119 TEXT Vector Search Examples.xlsx")
    
    # Load test data
    try:
        df = pd.read_excel(input_excel)
        logging.info(f"✅ Loaded {len(df)} rows from Excel file")
    except Exception as e:
        raise RuntimeError(f"❌ Error reading Excel file: {e}")

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
            raise ConnectionError("Elasticsearch cluster not reachable")
        logging.info("✅ Connected to Elasticsearch successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to connect to Elasticsearch: {e}")

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
        
        if expected_ecodes:
            ground_truth[search_term] = expected_ecodes

    logging.info(f"✅ Prepared {len(ground_truth)} test queries")

    # Run quantitative comparison only
    metrics, comparison = quantitative_comparison(es, ground_truth)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print("   📄 quantitative_e5_detailed.csv")
    print("   📄 quantitative_elser_detailed.csv")
    print("   📄 quantitative_comparison_summary.csv")
    print("\nNext Steps:")
    print("   1. Review detailed CSV files for per-query analysis")
    print("   2. Identify query patterns where each model excels")
    print("   3. Consider hybrid approach if performance is similar")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
