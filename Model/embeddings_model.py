from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import Union, List, Optional

class EmbeddingAgent:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the Embedding Agent with a SentenceTransformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            print(f"[EmbeddingAgent] Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"[EmbeddingAgent] Error loading model {model_name}: {e}")
            raise
    
    def get_embedding(self, text: Union[str, List[str]], 
                     convert_to_tensor: bool = True,
                     normalize_embeddings: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embeddings for text input.
        
        Args:
            text: Single text string or list of text strings
            convert_to_tensor: Whether to return torch tensors (True) or numpy arrays (False)
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            
        Returns:
            Embeddings as torch.Tensor or numpy.ndarray
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        
        try:
            embeddings = self.model.encode(
                text, 
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings
            )
            return embeddings
        except Exception as e:
            print(f"[EmbeddingAgent] Error generating embeddings: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], 
                           batch_size: int = 32,
                           convert_to_tensor: bool = True,
                           normalize_embeddings: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embeddings for a batch of texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            convert_to_tensor: Whether to return torch tensors
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Batch embeddings as torch.Tensor or numpy.ndarray
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings
            )
            return embeddings
        except Exception as e:
            print(f"[EmbeddingAgent] Error generating batch embeddings: {e}")
            raise
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            embeddings = self.get_embeddings_batch(
                [text1, text2], 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            similarity = torch.cosine_similarity(
                embeddings[0].unsqueeze(0), 
                embeddings[1].unsqueeze(0)
            ).item()
            return similarity
        except Exception as e:
            print(f"[EmbeddingAgent] Error computing similarity: {e}")
            raise
    
    def find_most_similar(self, query: str, candidates: List[str], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar texts to a query from a list of candidates.
        
        Args:
            query: Query text string
            candidates: List of candidate text strings to compare against
            top_k: Number of top similar candidates to return
            
        Returns:
            List of tuples (candidate_text, similarity_score, index) sorted by similarity
        """
        if not candidates:
            return []
        
        try:
            # Get embeddings for query and all candidates
            all_texts = [query] + candidates
            embeddings = self.get_embeddings_batch(
                all_texts, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            
            query_embedding = embeddings[0].unsqueeze(0)
            candidate_embeddings = embeddings[1:]
            
            # Compute similarities
            similarities = torch.cosine_similarity(
                query_embedding, 
                candidate_embeddings
            )
            
            # Get top-k most similar
            top_k = min(top_k, len(candidates))
            top_indices = torch.topk(similarities, top_k).indices
            
            results = []
            for idx in top_indices:
                idx_val = idx.item()
                similarity_score = similarities[idx_val].item()
                results.append((candidates[idx_val], similarity_score, idx_val))
            
            return results
            
        except Exception as e:
            print(f"[EmbeddingAgent] Error finding most similar texts: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        try:
            return {
                "model_name": self.model_name,
                "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown'),
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "device": str(self.model.device) if hasattr(self.model, 'device') else 'Unknown'
            }
        except Exception as e:
            print(f"[EmbeddingAgent] Error getting model info: {e}")
            return {"model_name": self.model_name, "error": str(e)}