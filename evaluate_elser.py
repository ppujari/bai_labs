import pandas as pd
import logging
from elasticsearch import Elasticsearch
import numpy as np
from sklearn.metrics import ndcg_score
import os
from collections import defaultdict

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
        "field": "content_embedding",  # ELSER uses this field
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
    
    for search_term in ground_truth.keys():
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
    
    return df, avg_metrics, results

def categorize_queries(queries):
    """
    Categorize queries by type for qualitative analysis
    """
    categories = {
        'Branded': [],
        'Descriptive': [],
        'Technical': [],
        'Generic': []
    }
    
    brands = ['nike', 'adidas', 'under armour', 'puma', 'reebok', 'new balance', 
              'columbia', 'north face', 'patagonia', 'carhartt', 'dickies']
    
    technical_terms = ['steel toe', 'waterproof', 'insulated', 'slip-resistant', 
                       'breathable', 'moisture-wicking', 'gore-tex']
    
    for query in queries:
        query_lower = query.lower()
        
        # Check if branded
        if any(brand in query_lower for brand in brands):
            categories['Branded'].append(query)
        # Check if technical
        elif any(term in query_lower for term in technical_terms):
            categories['Technical'].append(query)
        # Check if descriptive (contains adjectives/descriptors)
        elif len(query.split()) >= 3:
            categories['Descriptive'].append(query)
        else:
            categories['Generic'].append(query)
    
    return categories

def calculate_category_metrics(results, ground_truth, queries, k=32):
    """
    Calculate average metrics for a set of queries
    """
    metrics = []
    for query in queries:
        if query not in results:
            continue
        
        preds = results[query]
        true = ground_truth.get(query, [])
        if not true:
            continue
        
        p = precision_at_k(true, preds, k)
        r = recall_at_k(true, preds, k)
        mrr = reciprocal_rank(true, preds)
        ndcg = ndcg_at_k(true, preds, k)
        metrics.append((p, r, mrr, ndcg))
    
    if not metrics:
        return {'P': 0, 'R': 0, 'MRR': 0, 'NDCG': 0}
    
    avg = {
        'P': np.mean([m[0] for m in metrics]),
        'R': np.mean([m[1] for m in metrics]),
        'MRR': np.mean([m[2] for m in metrics]),
        'NDCG': np.mean([m[3] for m in metrics])
    }
    return avg

def find_example_failures(results_e5, results_elser, ground_truth):
    """
    Find example queries where one model significantly outperforms the other
    """
    examples = {
        'ELSER_better': [],
        'E5_better': [],
        'Both_fail': []
    }
    
    for query in ground_truth.keys():
        if query not in results_e5 or query not in results_elser:
            continue
        
        true = ground_truth[query]
        pred_e5 = results_e5[query]
        pred_elser = results_elser[query]
        
        # Find rank of first relevant result
        rank_e5 = None
        for i, pred in enumerate(pred_e5, 1):
            if pred in true:
                rank_e5 = i
                break
        
        rank_elser = None
        for i, pred in enumerate(pred_elser, 1):
            if pred in true:
                rank_elser = i
                break
        
        # Categorize
        if rank_elser and not rank_e5:
            examples['ELSER_better'].append((query, rank_elser, true[0] if true else "N/A"))
        elif rank_e5 and not rank_elser:
            examples['E5_better'].append((query, rank_e5, true[0] if true else "N/A"))
        elif not rank_e5 and not rank_elser:
            examples['Both_fail'].append((query, true[0] if true else "N/A"))
        elif rank_elser and rank_e5 and abs(rank_elser - rank_e5) >= 10:
            if rank_elser < rank_e5:
                examples['ELSER_better'].append((query, rank_elser, true[0] if true else "N/A"))
            else:
                examples['E5_better'].append((query, rank_e5, true[0] if true else "N/A"))
    
    return examples

def comprehensive_comparison(es, ground_truth):
    """
    Complete comparison framework
    """
    print("\n" + "="*80)
    print("E5 vs ELSER COMPREHENSIVE COMPARISON")
    print("="*80)
    
    all_results = {}
    all_metrics = {}
    all_dataframes = {}
    
    # Evaluate each model
    for model_name, model_config in MODELS.items():
        print(f"\n🔍 Evaluating {model_name}...")
        print(f"   Type: {model_config['type']}")
        print(f"   Index: {model_config['index']}")
        
        df, avg_metrics, results = evaluate_model(
            es, model_name, model_config, ground_truth, k=SIZE
        )
        
        all_results[model_name] = results
        all_metrics[model_name] = avg_metrics
        all_dataframes[model_name] = df
        
        # Save detailed results
        output_file = f"comparison_{model_name.lower()}_detailed.csv"
        df.to_csv(output_file, index=False)
        print(f"   ✅ Detailed results saved to {output_file}")
    
    # ========================================================================
    # QUANTITATIVE COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("📊 QUANTITATIVE METRICS")
    print("="*80)
    
    comparison_df = pd.DataFrame(all_metrics).T
    print("\n" + comparison_df.to_string())
    comparison_df.to_csv("model_comparison_summary.csv")
    
    # Determine winner by metric
    print("\n" + "="*80)
    print("🏆 WINNER BY METRIC")
    print("="*80)
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        improvement = (best_score / comparison_df[metric].min() - 1) * 100 if comparison_df[metric].min() > 0 else 0
        print(f"   {metric:12s}: {best_model:6s} ({best_score:.3f}) - {improvement:+.1f}% better")
    
    # ========================================================================
    # QUALITATIVE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("🔍 QUALITATIVE ANALYSIS BY QUERY TYPE")
    print("="*80)
    
    # Categorize queries
    query_categories = categorize_queries(list(ground_truth.keys()))
    
    category_results = defaultdict(dict)
    
    for category, queries in query_categories.items():
        if not queries:
            continue
        
        print(f"\n📁 {category.upper()} QUERIES ({len(queries)} queries)")
        print("-" * 80)
        
        for model_name in MODELS.keys():
            metrics = calculate_category_metrics(
                all_results[model_name], 
                ground_truth, 
                queries, 
                k=SIZE
            )
            category_results[category][model_name] = metrics
            
            print(f"   {model_name:6s}: P@{SIZE}={metrics['P']:.3f}, "
                  f"R@{SIZE}={metrics['R']:.3f}, "
                  f"MRR={metrics['MRR']:.3f}, "
                  f"NDCG@{SIZE}={metrics['NDCG']:.3f}")
        
        # Determine winner for this category
        best_model = max(
            category_results[category].items(),
            key=lambda x: x[1]['NDCG']
        )[0]
        print(f"   → {best_model} performs better for {category} queries")
    
    # ========================================================================
    # EXAMPLE FAILURES
    # ========================================================================
    print("\n" + "="*80)
    print("📝 EXAMPLE QUERY ANALYSIS")
    print("="*80)
    
    examples = find_example_failures(
        all_results['E5'],
        all_results['ELSER'],
        ground_truth
    )
    
    # Show ELSER wins
    print("\n✅ QUERIES WHERE ELSER SIGNIFICANTLY OUTPERFORMS E5:")
    for i, (query, rank, expected_id) in enumerate(examples['ELSER_better'][:5], 1):
        print(f"   {i}. '{query}'")
        print(f"      ELSER found at rank {rank}, E5 missed")
        print(f"      Expected: {expected_id}")
    
    # Show E5 wins
    print("\n✅ QUERIES WHERE E5 SIGNIFICANTLY OUTPERFORMS ELSER:")
    for i, (query, rank, expected_id) in enumerate(examples['E5_better'][:5], 1):
        print(f"   {i}. '{query}'")
        print(f"      E5 found at rank {rank}, ELSER missed")
        print(f"      Expected: {expected_id}")
    
    # Show both fail
    print("\n❌ QUERIES WHERE BOTH MODELS FAIL:")
    for i, (query, expected_id) in enumerate(examples['Both_fail'][:5], 1):
        print(f"   {i}. '{query}'")
        print(f"      Expected: {expected_id}")
    
    # ========================================================================
    # INTERPRETATION
    # ========================================================================
    print("\n" + "="*80)
    print("💡 INTERPRETATION")
    print("="*80)
    
    e5_ndcg = all_metrics['E5'][f'NDCG@{SIZE}']
    elser_ndcg = all_metrics['ELSER'][f'NDCG@{SIZE}']
    
    if elser_ndcg > e5_ndcg:
        improvement = ((elser_ndcg / e5_ndcg) - 1) * 100
        print(f"\n✅ ELSER shows {improvement:.1f}% higher NDCG than E5 for this dataset")
    else:
        improvement = ((e5_ndcg / elser_ndcg) - 1) * 100
        print(f"\n✅ E5 shows {improvement:.1f}% higher NDCG than ELSER for this dataset")
    
    # Check if either is acceptable
    acceptable_threshold = 0.40
    if max(e5_ndcg, elser_ndcg) < acceptable_threshold:
        print(f"⚠️  Both models perform below acceptable threshold (NDCG < {acceptable_threshold})")
        print("   Consider: Hybrid search, better document text, or query expansion")
    
    print("\n⚠️  Different architectures make direct comparison challenging:")
    print("   - E5 (Dense): Semantic understanding through vector similarity")
    print("   - ELSER (Sparse): Learned lexical expansion with term weights")
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if elser_ndcg > e5_ndcg * 1.2:  # ELSER 20% better
        print("\n✅ ELSER significantly outperforms E5 for your use case")
        print("   Recommendation: Use ELSER as primary search method")
    elif e5_ndcg > elser_ndcg * 1.2:  # E5 20% better
        print("\n✅ E5 significantly outperforms ELSER for your use case")
        print("   Recommendation: Use E5 as primary search method")
    else:
        print("\n⚖️  Performance is comparable between models")
        print("   Recommendation: Consider hybrid approach combining both")
    
    print("\n🔄 Consider implementing:")
    print("   1. Hybrid search (E5 + ELSER + keyword)")
    print("   2. Query expansion with synonyms")
    print("   3. Improve document descriptions")
    print("   4. Re-ranking with cross-encoder")
    
    return all_results, all_metrics, comparison_df

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

    # Run comprehensive comparison
    results, metrics, comparison = comprehensive_comparison(es, ground_truth)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("   - comparison_e5_detailed.csv")
    print("   - comparison_elser_detailed.csv")
    print("   - model_comparison_summary.csv")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
