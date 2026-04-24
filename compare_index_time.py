import time
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import os
from datetime import datetime

# Configuration
ES_HOST = "https://dsg-search-dev-east.es.eastus.azure.elastic-cloud.com"
ES_USER = os.getenv("ES_USER", "data_ingestion")
ES_PASS = os.getenv("ES_PASS", "FVG8%7SMUys$dp4m")

# Test configuration
NUM_DOCS = 100
TEST_INDEXES = {
    "E5": "pp-e5-test",
    "ELSER": "pp-elser-test"
}

# Model configurations
MODELS = {
    "E5": {
        "model_id": "intfloat__e5-small",
        "field": "content_embedding",
        "dims": 384
    },
    "ELSER": {
        "model_id": ".elser_model_2",
        "field": "content_embedding"
    }
}

# Sample product data for testing
SAMPLE_PRODUCTS = [
    "Nike Air Max running shoes with cushioned sole and breathable mesh upper",
    "Adidas Ultraboost performance sneakers for marathon training",
    "Under Armour moisture-wicking athletic shirt for workout",
    "Columbia waterproof hiking boots with ankle support",
    "North Face insulated winter jacket with down filling",
    "Carhartt durable work pants with reinforced knees",
    "Patagonia fleece pullover for outdoor activities",
    "Reebok CrossFit training shoes with stable platform",
    "New Balance trail running shoes with aggressive tread",
    "Puma soccer cleats with synthetic leather upper",
    "Brooks Ghost running shoes for neutral runners",
    "ASICS gel-cushioned tennis shoes for court sports",
    "Timberland steel toe work boots for construction",
    "Dickies work shirt with moisture management",
    "Champion reverse weave hoodie with kangaroo pocket",
    "Levi's denim jeans with classic five-pocket design",
    "Wrangler cargo pants with multiple storage pockets",
    "Hanes cotton t-shirt pack for everyday wear",
    "Fruit of the Loom crew neck sweatshirt",
    "Gildan heavy blend fleece for screen printing",
]

def generate_test_documents(num_docs):
    """Generate test documents for indexing"""
    docs = []
    for i in range(num_docs):
        # Cycle through sample products
        product_desc = SAMPLE_PRODUCTS[i % len(SAMPLE_PRODUCTS)]
        
        doc = {
            "_id": f"TEST-DOC-{i+1:05d}",
            "title": f"Test Product {i+1}",
            "description": f"{product_desc}. Product ID: {i+1}",
            "content": f"passage: {product_desc}",  # For embedding
            "price": 49.99 + (i % 50),
            "category": ["Apparel", "Footwear", "Equipment"][i % 3],
            "timestamp": datetime.now().isoformat()
        }
        docs.append(doc)
    
    return docs

def create_e5_index(es, index_name):
    """Create index with E5 dense vector configuration"""
    index_config = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.default_pipeline": "e5-embedding-pipeline"
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "description": {"type": "text"},
                "content": {"type": "text"},
                "price": {"type": "float"},
                "category": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "content_embedding": {
                    "type": "dense_vector",
                    "dims": MODELS["E5"]["dims"],
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    # Create inference pipeline for E5
    pipeline_config = {
        "description": "E5 embedding pipeline for testing",
        "processors": [
            {
                "inference": {
                    "model_id": MODELS["E5"]["model_id"],
                    "input_output": {
                        "input_field": "content",
                        "output_field": "content_embedding"
                    }
                }
            }
        ]
    }
    
    try:
        # Create pipeline
        es.ingest.put_pipeline(id="e5-embedding-pipeline", body=pipeline_config)
        print(f"   ✅ Created E5 inference pipeline")
        
        # Create index
        es.indices.create(index=index_name, body=index_config)
        print(f"   ✅ Created E5 index: {index_name}")
        
    except Exception as e:
        print(f"   ⚠️  Error creating E5 index: {e}")
        raise

def create_elser_index(es, index_name):
    """Create index with ELSER sparse vector configuration"""
    index_config = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.default_pipeline": "elser-embedding-pipeline"
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "description": {"type": "text"},
                "content": {"type": "text"},
                "price": {"type": "float"},
                "category": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "content_embedding": {
                    "type": "sparse_vector"
                }
            }
        }
    }
    
    # Create inference pipeline for ELSER
    pipeline_config = {
        "description": "ELSER embedding pipeline for testing",
        "processors": [
            {
                "inference": {
                    "model_id": MODELS["ELSER"]["model_id"],
                    "input_output": {
                        "input_field": "content",
                        "output_field": "content_embedding"
                    }
                }
            }
        ]
    }
    
    try:
        # Create pipeline
        es.ingest.put_pipeline(id="elser-embedding-pipeline", body=pipeline_config)
        print(f"   ✅ Created ELSER inference pipeline")
        
        # Create index
        es.indices.create(index=index_name, body=index_config)
        print(f"   ✅ Created ELSER index: {index_name}")
        
    except Exception as e:
        print(f"   ⚠️  Error creating ELSER index: {e}")
        raise

def index_documents(es, index_name, documents):
    """Index documents and return time taken"""
    
    def doc_generator():
        for doc in documents:
            yield {
                "_index": index_name,
                "_id": doc["_id"],
                "_source": doc
            }
    
    start_time = time.time()
    
    try:
        success, failed = bulk(
            es,
            doc_generator(),
            stats_only=False,
            raise_on_error=False,
            request_timeout=300
        )
        
        # Wait for indexing to complete
        es.indices.refresh(index=index_name)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            "success": success,
            "failed": len(failed) if failed else 0,
            "time_seconds": elapsed_time,
            "docs_per_second": success / elapsed_time if elapsed_time > 0 else 0
        }
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"   ❌ Indexing error: {e}")
        return {
            "success": 0,
            "failed": len(documents),
            "time_seconds": elapsed_time,
            "docs_per_second": 0,
            "error": str(e)
        }

def delete_test_indexes(es, indexes):
    """Delete test indexes and pipelines"""
    print("\n" + "="*80)
    print("🗑️  CLEANUP: Deleting Test Indexes")
    print("="*80)
    
    # Delete indexes
    for model_name, index_name in indexes.items():
        try:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
                print(f"   ✅ Deleted index: {index_name}")
            else:
                print(f"   ⚠️  Index not found: {index_name}")
        except Exception as e:
            print(f"   ❌ Error deleting index {index_name}: {e}")
    
    # Delete pipelines
    pipelines = ["e5-embedding-pipeline", "elser-embedding-pipeline"]
    for pipeline in pipelines:
        try:
            es.ingest.delete_pipeline(id=pipeline)
            print(f"   ✅ Deleted pipeline: {pipeline}")
        except Exception as e:
            print(f"   ⚠️  Pipeline deletion: {e}")

def run_indexing_comparison():
    """Main comparison function"""
    
    print("\n" + "="*80)
    print("⚡ INDEXING PERFORMANCE COMPARISON: E5 vs ELSER")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"   Documents to index: {NUM_DOCS}")
    print(f"   E5 Index: {TEST_INDEXES['E5']}")
    print(f"   ELSER Index: {TEST_INDEXES['ELSER']}")
    
    # Connect to Elasticsearch
    try:
        es = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USER, ES_PASS),
            request_timeout=300,
            max_retries=3,
            retry_on_timeout=True
        )
        if not es.ping():
            raise ConnectionError("Cannot connect to Elasticsearch")
        print(f"\n✅ Connected to Elasticsearch")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return
    
    # Generate test documents
    print(f"\n📝 Generating {NUM_DOCS} test documents...")
    test_docs = generate_test_documents(NUM_DOCS)
    print(f"   ✅ Generated {len(test_docs)} documents")
    
    results = {}
    
    # ========================================================================
    # TEST E5 INDEXING
    # ========================================================================
    print("\n" + "="*80)
    print("🔍 Testing E5 (Dense Vector) Indexing")
    print("="*80)
    
    try:
        # Create E5 index
        create_e5_index(es, TEST_INDEXES["E5"])
        
        # Index documents
        print(f"\n   ⏱️  Indexing {NUM_DOCS} documents with E5 embeddings...")
        e5_results = index_documents(es, TEST_INDEXES["E5"], test_docs)
        results["E5"] = e5_results
        
        print(f"\n   📊 E5 Results:")
        print(f"      Success: {e5_results['success']}/{NUM_DOCS} documents")
        print(f"      Failed: {e5_results['failed']} documents")
        print(f"      Time: {e5_results['time_seconds']:.2f} seconds")
        print(f"      Throughput: {e5_results['docs_per_second']:.2f} docs/sec")
        
        if e5_results.get('error'):
            print(f"      Error: {e5_results['error']}")
        
    except Exception as e:
        print(f"\n   ❌ E5 test failed: {e}")
        results["E5"] = {"error": str(e)}
    
    # Small delay between tests
    time.sleep(2)
    
    # ========================================================================
    # TEST ELSER INDEXING
    # ========================================================================
    print("\n" + "="*80)
    print("🎯 Testing ELSER (Sparse Vector) Indexing")
    print("="*80)
    
    try:
        # Create ELSER index
        create_elser_index(es, TEST_INDEXES["ELSER"])
        
        # Index documents
        print(f"\n   ⏱️  Indexing {NUM_DOCS} documents with ELSER embeddings...")
        elser_results = index_documents(es, TEST_INDEXES["ELSER"], test_docs)
        results["ELSER"] = elser_results
        
        print(f"\n   📊 ELSER Results:")
        print(f"      Success: {elser_results['success']}/{NUM_DOCS} documents")
        print(f"      Failed: {elser_results['failed']} documents")
        print(f"      Time: {elser_results['time_seconds']:.2f} seconds")
        print(f"      Throughput: {elser_results['docs_per_second']:.2f} docs/sec")
        
        if elser_results.get('error'):
            print(f"      Error: {elser_results['error']}")
        
    except Exception as e:
        print(f"\n   ❌ ELSER test failed: {e}")
        results["ELSER"] = {"error": str(e)}
    
    # ========================================================================
    # COMPARISON REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("📊 INDEXING PERFORMANCE COMPARISON")
    print("="*80)
    
    if "E5" in results and "ELSER" in results:
        e5_time = results["E5"].get("time_seconds", 0)
        elser_time = results["ELSER"].get("time_seconds", 0)
        
        print(f"\n┌{'─'*76}┐")
        print(f"│ {'Metric':<30} │ {'E5 (Dense)':<20} │ {'ELSER (Sparse)':<20} │")
        print(f"├{'─'*76}┤")
        print(f"│ {'Total Time':<30} │ {e5_time:>17.2f}s │ {elser_time:>17.2f}s │")
        print(f"│ {'Throughput':<30} │ {results['E5'].get('docs_per_second', 0):>14.2f} d/s │ {results['ELSER'].get('docs_per_second', 0):>14.2f} d/s │")
        print(f"│ {'Success Rate':<30} │ {results['E5'].get('success', 0)/NUM_DOCS*100:>15.1f}% │ {results['ELSER'].get('success', 0)/NUM_DOCS*100:>15.1f}% │")
        print(f"│ {'Avg Time per Doc':<30} │ {e5_time/NUM_DOCS*1000:>14.0f} ms │ {elser_time/NUM_DOCS*1000:>14.0f} ms │")
        print(f"└{'─'*76}┘")
        
        # Determine winner
        print("\n" + "="*80)
        print("🏆 PERFORMANCE ANALYSIS")
        print("="*80)
        
        if e5_time > 0 and elser_time > 0:
            if e5_time < elser_time:
                speedup = (elser_time / e5_time - 1) * 100
                print(f"\n✅ E5 is FASTER")
                print(f"   E5 completed {speedup:.1f}% faster than ELSER")
                print(f"   Time difference: {elser_time - e5_time:.2f} seconds")
            else:
                speedup = (e5_time / elser_time - 1) * 100
                print(f"\n✅ ELSER is FASTER")
                print(f"   ELSER completed {speedup:.1f}% faster than E5")
                print(f"   Time difference: {e5_time - elser_time:.2f} seconds")
        
        # Additional insights
        print("\n💡 Insights:")
        print(f"   • E5 generates {MODELS['E5']['dims']}-dimensional dense vectors")
        print(f"   • ELSER generates ~4000-dimensional sparse vectors")
        print(f"   • E5 embeddings: ~{MODELS['E5']['dims'] * 4 / 1024:.1f} KB per document")
        print(f"   • ELSER embeddings: ~1-2 KB per document (sparse)")
        
        if e5_time < elser_time:
            print(f"\n   E5 is faster for indexing, but consider:")
            print(f"   - Search quality differences")
            print(f"   - Index size (E5 may be larger)")
            print(f"   - Query speed at search time")
        else:
            print(f"\n   ELSER is faster for indexing, but consider:")
            print(f"   - Search quality differences")
            print(f"   - Computational complexity")
            print(f"   - Query speed at search time")
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    delete_test_indexes(es, TEST_INDEXES)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise, show only warnings/errors
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        run_indexing_comparison()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        logging.exception("Detailed error:")
```

## **Expected Output:**
```
================================================================================
⚡ INDEXING PERFORMANCE COMPARISON: E5 vs ELSER
================================================================================

Test Configuration:
   Documents to index: 100
   E5 Index: pp-e5-test
   ELSER Index: pp-elser-test

✅ Connected to Elasticsearch

📝 Generating 100 test documents...
   ✅ Generated 100 documents

================================================================================
🔍 Testing E5 (Dense Vector) Indexing
================================================================================
   ✅ Created E5 inference pipeline
   ✅ Created E5 index: pp-e5-test

   ⏱️  Indexing 100 documents with E5 embeddings...

   📊 E5 Results:
      Success: 100/100 documents
      Failed: 0 documents
      Time: 12.45 seconds
      Throughput: 8.03 docs/sec

================================================================================
🎯 Testing ELSER (Sparse Vector) Indexing
================================================================================
   ✅ Created ELSER inference pipeline
   ✅ Created ELSER index: pp-elser-test

   ⏱️  Indexing 100 documents with ELSER embeddings...

   📊 ELSER Results:
      Success: 100/100 documents
      Failed: 0 documents
      Time: 18.72 seconds
      Throughput: 5.34 docs/sec

================================================================================
📊 INDEXING PERFORMANCE COMPARISON
================================================================================

┌────────────────────────────────────────────────────────────────────────────┐
│ Metric                         │ E5 (Dense)           │ ELSER (Sparse)       │
├────────────────────────────────────────────────────────────────────────────┤
│ Total Time                     │            12.45s │            18.72s │
│ Throughput                     │           8.03 d/s │           5.34 d/s │
│ Success Rate                   │             100.0% │             100.0% │
│ Avg Time per Doc               │             124 ms │             187 ms │
└────────────────────────────────────────────────────────────────────────────┘

================================================================================
🏆 PERFORMANCE ANALYSIS
================================================================================

✅ E5 is FASTER
   E5 completed 50.4% faster than ELSER
   Time difference: 6.27 seconds

💡 Insights:
   • E5 generates 384-dimensional dense vectors
   • ELSER generates ~4000-dimensional sparse vectors
   • E5 embeddings: ~1.5 KB per document
   • ELSER embeddings: ~1-2 KB per document (sparse)

   E5 is faster for indexing, but consider:
   - Search quality differences
   - Index size (E5 may be larger)
   - Query speed at search time

================================================================================
🗑️  CLEANUP: Deleting Test Indexes
================================================================================
   ✅ Deleted index: pp-e5-test
   ✅ Deleted index: pp-elser-test
   ✅ Deleted pipeline: e5-embedding-pipeline
   ✅ Deleted pipeline: elser-embedding-pipeline

================================================================================
✅ COMPARISON COMPLETE
================================================================================
