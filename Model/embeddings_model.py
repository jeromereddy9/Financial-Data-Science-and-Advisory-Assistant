from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import Union, List, Optional

class EmbeddingAgent:
    """
    Embedding agent that converts text into high-dimensional vectors for semantic similarity.
    Uses SentenceTransformer models to create 384-dimensional embeddings (for all-MiniLM-L6-v2).
    Core component for memory search and semantic matching in the financial advisory system.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the Embedding Agent with a SentenceTransformer model.
        Default model (all-MiniLM-L6-v2) is lightweight, fast, and provides good quality embeddings.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
                             Default: "all-MiniLM-L6-v2" (384 dims, 80MB, good speed/quality balance)
                             Alternatives: "all-mpnet-base-v2" (768 dims, better quality but slower)
        """
        self.model_name = model_name
        try:
            # Load pre-trained sentence transformer model
            # Model automatically downloads on first use and caches locally
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
        Core method for converting text to semantic vectors.
        
        Args:
            text: Single text string or list of text strings to embed
            convert_to_tensor: Whether to return torch tensors (True) or numpy arrays (False)
                              Tensors are better for GPU operations, arrays for storage
            normalize_embeddings: Whether to normalize embeddings to unit vectors (length 1)
                                 Normalization makes cosine similarity equivalent to dot product
            
        Returns:
            Embeddings as torch.Tensor or numpy.ndarray
            Shape: (embedding_dim,) for single text or (batch_size, embedding_dim) for list
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        
        try:
            # Encode text into semantic vector representation
            # Model handles tokenization, encoding, and pooling internally
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
        Processes multiple texts in parallel for better performance.
        
        Args:
            texts: List of text strings to embed (e.g., multiple queries or documents)
            batch_size: Number of texts to process at once (larger = faster but more memory)
                       Default 32 is good balance for most use cases
            convert_to_tensor: Whether to return torch tensors (GPU-friendly)
            normalize_embeddings: Whether to normalize to unit vectors (better for cosine similarity)
            
        Returns:
            Batch embeddings as torch.Tensor or numpy.ndarray
            Shape: (num_texts, embedding_dim) - one vector per input text
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        try:
            # Batch encoding is more efficient than encoding texts individually
            # Model automatically handles padding and batching
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
        Used for measuring semantic similarity between queries and documents.
        
        Args:
            text1: First text string (e.g., user query)
            text2: Second text string (e.g., stored session)
            
        Returns:
            Cosine similarity score between -1 and 1
            1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite meaning
            In practice, values > 0.5 indicate strong similarity
        """
        try:
            # Generate normalized embeddings for both texts
            # Normalization makes cosine similarity calculation simpler
            embeddings = self.get_embeddings_batch(
                [text1, text2], 
                convert_to_tensor=True, 
                normalize_embeddings=True  # Critical for accurate cosine similarity
            )
            
            # Compute cosine similarity using PyTorch
            # unsqueeze(0) adds batch dimension required by cosine_similarity
            similarity = torch.cosine_similarity(
                embeddings[0].unsqueeze(0), 
                embeddings[1].unsqueeze(0)
            ).item()  # Convert tensor to Python float
            return similarity
        except Exception as e:
            print(f"[EmbeddingAgent] Error computing similarity: {e}")
            raise
    
    def find_most_similar(self, query: str, candidates: List[str], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar texts to a query from a list of candidates.
        Core method for semantic search - finds relevant past sessions or documents.
        
        Args:
            query: Query text string (e.g., "What's the performance of MTN stock?")
            candidates: List of candidate text strings to compare against (e.g., past queries)
            top_k: Number of top similar candidates to return (default 5)
            
        Returns:
            List of tuples (candidate_text, similarity_score, index) sorted by similarity descending
            Example: [("MTN stock analysis", 0.87, 3), ("MTN price trends", 0.82, 7), ...]
        """
        if not candidates:
            return []
        
        try:
            # Get embeddings for query and all candidates in one batch
            # More efficient than encoding separately
            all_texts = [query] + candidates
            embeddings = self.get_embeddings_batch(
                all_texts, 
                convert_to_tensor=True, 
                normalize_embeddings=True  # Essential for fair similarity comparison
            )
            
            # Separate query embedding from candidate embeddings
            query_embedding = embeddings[0].unsqueeze(0)  # Shape: (1, embedding_dim)
            candidate_embeddings = embeddings[1:]         # Shape: (num_candidates, embedding_dim)
            
            # Compute cosine similarity between query and all candidates at once
            # Result shape: (num_candidates,) - one score per candidate
            similarities = torch.cosine_similarity(
                query_embedding, 
                candidate_embeddings
            )
            
            # Get indices of top-k highest similarity scores
            top_k = min(top_k, len(candidates))  # Ensure we don't request more than available
            top_indices = torch.topk(similarities, top_k).indices
            
            # Build result list with candidate text, score, and original index
            results = []
            for idx in top_indices:
                idx_val = idx.item()  # Convert tensor to Python int
                similarity_score = similarities[idx_val].item()  # Convert tensor to Python float
                results.append((candidates[idx_val], similarity_score, idx_val))
            
            return results
            
        except Exception as e:
            print(f"[EmbeddingAgent] Error finding most similar texts: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        Useful for debugging and system introspection.
        
        Returns:
            Dictionary with model information:
            - model_name: Name of the model
            - max_seq_length: Maximum token length (usually 128-512)
            - embedding_dimension: Size of output vectors (384 for all-MiniLM-L6-v2)
            - device: Where model runs (cuda:0 for GPU, cpu for CPU)
        """
        try:
            return {
                "model_name": self.model_name,
                # Maximum sequence length in tokens (text longer than this gets truncated)
                "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown'),
                # Dimension of embedding vectors (384 for MiniLM, 768 for MPNet)
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                # Device where model is loaded (GPU if available for faster processing)
                "device": str(self.model.device) if hasattr(self.model, 'device') else 'Unknown'
            }
        except Exception as e:
            print(f"[EmbeddingAgent] Error getting model info: {e}")
            return {"model_name": self.model_name, "error": str(e)}