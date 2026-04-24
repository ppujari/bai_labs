import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class EmbeddingCollapseTest:
    """
    Args:
        model_name: E5 model variant
            - "intfloat/e5-small-v2" (33M params, fast)
            - "intfloat/e5-base-v2" (109M params, better quality)
            - "intfloat/e5-large-v2" (335M params, best quality)
        normalize_embeddings: Whether to L2 normalize output
    """
    def __init__(self, embeddings, product_types, descriptions, model_name):
        self.embeddings = np.array(embeddings)
        self.product_types = product_types
        self.descriptions = descriptions
        self.similarity_matrix = cosine_similarity(self.embeddings)

        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = True
        
        # E5 embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
def embed_passages(self, texts, add_prefix=True):
        """
        Embed product descriptions (passages for retrieval)
        Args:
            texts: List of product descriptions
            add_prefix: Add "passage: " prefix (recommended for E5)
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        
        # Add prefix for better retrieval performance
        if add_prefix:
            texts_to_embed = [f"passage: {text}" for text in texts]
        else:
            texts_to_embed = texts
        
        # Encode all texts at once
        embeddings = self.model.encode(
            texts_to_embed,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings
    
    def embed_query(self, text, add_prefix=True):
        """
        Embed a search query
        Args:
            text: Query text
            add_prefix: Add "query: " prefix (recommended for E5)
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        
        if add_prefix:
            text_to_embed = f"query: {text}"
        else:
            text_to_embed = text
        
        embedding = self.model.encode(
            text_to_embed,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embedding
    
    def test_silhouette_score(self):
        """Calculate silhouette score for product type clustering"""
        labels = [hash(t) % 256 for t in self.product_types]
        score = silhouette_score(self.embeddings, labels)
        return score
    
    def test_separation_ratio(self):
        """Calculate intra vs inter-class distance ratio"""
        intra_distances = []
        inter_distances = []
        
        for i in range(len(self.embeddings)):
            for j in range(i+1, len(self.embeddings)):
                sim = self.similarity_matrix[i][j]
                if self.product_types[i] == self.product_types[j]:
                    intra_distances.append(sim)
                else:
                    inter_distances.append(sim)
        
        intra_mean = np.mean(intra_distances)
        inter_mean = np.mean(inter_distances)
        ratio = (intra_mean - inter_mean) / inter_mean
        
        return {
            'intra_class_mean': intra_mean,
            'inter_class_mean': inter_mean,
            'separation_ratio': ratio
        }
    
    def test_retrieval_accuracy(self, k=10):
        """Verify correct product types in top-k neighbors"""
        correct = 0
        total = 0
        
        for i in range(len(self.embeddings)):
            distances = self.similarity_matrix[i]
            top_k = np.argsort(distances)[-k-1:-1][::-1]  # Exclude self
            
            same_type_count = sum(1 for idx in top_k if self.product_types[idx] == self.product_types[i])
            correct += same_type_count
            total += k
        
        accuracy = correct / total
        return accuracy
    
    def test_nearest_neighbors(self, sample_size=5):
        """Print nearest neighbors for random samples"""
        results = {}
        
        for _ in range(sample_size):
            idx = np.random.randint(len(self.embeddings))
            distances = self.similarity_matrix[idx]
            top_5 = np.argsort(distances)[-6:-1][::-1]
            
            results[f"{self.product_types[idx]}_{idx}"] = [
                {
                    'type': self.product_types[j],
                    'similarity': distances[j]
                }
                for j in top_5
            ]
        
        return results
    
    def run_all_tests(self):
        """Execute complete test suite"""
        return {
            'silhouette_score': self.test_silhouette_score(),
            'separation_metrics': self.test_separation_ratio(),
            'retrieval_accuracy': self.test_retrieval_accuracy(),
            'nearest_neighbors_sample': self.test_nearest_neighbors()
        }

# Usage:
# embeddings = [embed(desc) for desc in descriptions]
# tester = EmbeddingCollapseTest(embeddings, product_types, descriptions)
# results = tester.run_all_tests()
# print(json.dumps(results, indent=2))
