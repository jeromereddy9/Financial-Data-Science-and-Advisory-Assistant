from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class EmbeddingModel:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self.embeddings_cache = {}
        self.documents = []
    
    def encode_documents(self, documents):
        """
        Create embeddings for a list of documents
        """
        embeddings = self.model.encode(documents)
        self.documents.extend(documents)
        return embeddings
    
    def semantic_search(self, query, documents, top_k=5):
        """
        Perform semantic search on documents
        """
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Encode documents if not already done
        if isinstance(documents, list):
            doc_embeddings = self.model.encode(documents)
        else:
            doc_embeddings = documents
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx] if isinstance(documents, list) else self.documents[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        
        return results
    
    def build_document_index(self, documents, save_path="embeddings_index.pkl"):
        """
        Build and save document index
        """
        embeddings = self.encode_documents(documents)
        
        index_data = {
            'embeddings': embeddings,
            'documents': documents
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        return embeddings
    
    def load_document_index(self, load_path="embeddings_index.pkl"):
        """
        Load pre-built document index
        """
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            return index_data['embeddings']
        else:
            print(f"Index file {load_path} not found")
            return None
    
    def find_similar_context(self, query, context_documents, threshold=0.3):
        """
        Find relevant context for a query
        """
        results = self.semantic_search(query, context_documents, top_k=3)
        
        # Filter by similarity threshold
        relevant_context = [
            result['document'] for result in results 
            if result['similarity'] > threshold
        ]
        
        return relevant_context
