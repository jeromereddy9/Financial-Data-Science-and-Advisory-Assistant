import os
import pickle
import json
from collections import deque
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import hashlib

class MemoryManager:
    """
    Manages persistent storage and semantic retrieval of financial advisory sessions.
    Uses sentence embeddings to find similar past conversations and provide context.
    """
    
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", 
                 memory_file="financial_memory.pkl", max_sessions=200):
        """
        Initialize Memory Manager for financial advisory sessions.
        
        Args:
            embedding_model_name: SentenceTransformer model for embeddings (384-dim vectors)
            memory_file: File path for persistent storage using pickle
            max_sessions: Maximum number of sessions to store (uses deque for auto-rotation)
        """
        self.embedding_model_name = embedding_model_name
        self.memory_file = memory_file
        self.max_sessions = max_sessions
        
        try:
            # Load the sentence transformer model for creating embeddings
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"[MemoryManager] Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            print(f"[MemoryManager] Error loading embedding model: {e}")
            raise
        
        # Load existing memory from disk or initialize empty deque
        self.memory = self._load_memory()
        
        # Predefined categories with keywords for automatic session classification
        self.session_categories = {
            "portfolio_analysis": ["portfolio", "diversification", "allocation", "holdings"],
            "stock_analysis": ["stock", "share", "equity", "company", "JSE"],
            "market_insight": ["market", "economy", "sector", "trend", "outlook"],
            "risk_management": ["risk", "volatility", "hedge", "protection", "insurance"],
            "general_advice": ["invest", "advice", "recommend", "strategy", "plan"]
        }
    
    def add_session(self, query: str, advisor_response: str, web_context: Optional[List[Dict]] = None,
                   data_insights: Optional[str] = None, summary: Optional[str] = None,
                   user_metadata: Optional[Dict] = None) -> str:
        """
        Add a comprehensive financial advisory session to memory.
        Creates embeddings for semantic search and stores all session components.
        
        Args:
            query: Original user query
            advisor_response: Response from advisor agent
            web_context: Context from web supplementation agent (news articles, market data)
            data_insights: Insights from data analysis agent
            summary: Pre-generated summary (optional, will create basic one if missing)
            user_metadata: Additional metadata about the session (user preferences, etc.)
        
        Returns:
            Session ID for reference in future queries
        """
        try:
            # Create unique session identifier using MD5 hash
            session_id = self._generate_session_id(query, advisor_response)
            
            # Automatically categorize session based on content keywords
            session_category = self._categorize_session(query + " " + advisor_response)
            
            # Combine all session components into comprehensive text for embedding
            full_session_text = self._create_session_text(
                query, advisor_response, web_context, data_insights
            )
            
            # Generate three types of embeddings for different search strategies:
            # 1. Query embedding - for finding similar questions
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            # 2. Response embedding - for finding similar answers
            response_embedding = self.embedding_model.encode(advisor_response, convert_to_tensor=False)
            # 3. Full session embedding - for comprehensive similarity
            full_embedding = self.embedding_model.encode(full_session_text, convert_to_tensor=False)
            
            # Create comprehensive session data structure with all components
            session_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "advisor_response": advisor_response,
                "web_context": web_context or [],
                "data_insights": data_insights or "",
                "summary": summary or self._create_basic_summary(query, advisor_response),
                "category": session_category,
                "embeddings": {
                    "query": query_embedding,
                    "response": response_embedding,
                    "full_session": full_embedding
                },
                "metadata": {
                    "response_length": len(advisor_response),
                    "query_length": len(query),
                    "has_web_context": bool(web_context),
                    "has_data_insights": bool(data_insights),
                    **(user_metadata or {})  # Merge any additional user metadata
                }
            }
            
            # Add to memory (deque auto-removes oldest if at max_sessions)
            self.memory.append(session_data)
            # Persist to disk immediately
            self._save_memory()
            
            print(f"[MemoryManager] Added session {session_id} (category: {session_category})")
            return session_id
            
        except Exception as e:
            print(f"[MemoryManager] Error adding session: {e}")
            # Return fallback session ID to maintain system flow
            return "error_session_" + str(hash(query))[:8]
    
    def search_memory(self, query_text: str, top_k: int = 3, 
                     category_filter: Optional[str] = None,
                     recency_bias: float = 0.1,
                     similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search memory for relevant past sessions with advanced filtering.
        Uses semantic similarity + category matching + recency bias for ranking.
        
        Args:
            query_text: Current query to search for similar sessions
            top_k: Number of top similar sessions to return
            category_filter: Filter by session category (optional, e.g., "stock_analysis")
            recency_bias: Weight given to recent sessions (0-1, 0.1 = slight preference)
            similarity_threshold: Minimum similarity score to include (0-1, 0.3 = moderate)
        
        Returns:
            List of relevant session data sorted by relevance score
        """
        try:
            if not self.memory:
                return []
            
            # Create embedding for current query
            query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=False)
            # Determine query category for category boost
            query_category = self._categorize_session(query_text)
            
            scored_sessions = []
            
            for session in self.memory:
                # Calculate semantic similarity between current query and stored sessions
                # Compare against full session embedding for comprehensive match
                full_similarity = self._cosine_similarity(
                    query_embedding, session["embeddings"]["full_session"]
                )
                # Compare against stored query embedding for question-to-question match
                query_similarity = self._cosine_similarity(
                    query_embedding, session["embeddings"]["query"]
                )
                
                # Weighted combination: query similarity is more important (70/30 split)
                # This helps find sessions where user asked similar questions
                base_score = 0.7 * query_similarity + 0.3 * full_similarity
                
                # Boost score if categories match (10% bonus for same category)
                category_boost = 0.1 if session["category"] == query_category else 0.0
                
                # Apply recency bias (newer sessions get slight boost)
                recency_score = self._calculate_recency_score(session["timestamp"], recency_bias)
                
                # Combine all scoring factors for final relevance score
                final_score = base_score + category_boost + recency_score
                
                # Apply filters: only include if meets threshold and category filter
                if final_score >= similarity_threshold:
                    if category_filter is None or session["category"] == category_filter:
                        scored_sessions.append((final_score, session))
            
            # Sort by score descending (most relevant first) and return top_k
            scored_sessions.sort(key=lambda x: x[0], reverse=True)
            
            # Add relevance score to each result for transparency
            results = []
            for score, session in scored_sessions[:top_k]:
                result_session = session.copy()
                result_session["relevance_score"] = float(score)
                results.append(result_session)
            
            print(f"[MemoryManager] Found {len(results)} relevant sessions for query")
            return results
            
        except Exception as e:
            print(f"[MemoryManager] Error searching memory: {e}")
            return []
    
    def get_recent_sessions(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent sessions within specified days.
        Useful for showing conversation history or recent activity.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_sessions = []
            
            # Iterate in reverse order to get most recent first
            for session in reversed(self.memory):
                session_date = datetime.fromisoformat(session["timestamp"])
                if session_date >= cutoff_date:
                    recent_sessions.append(session)
                    if len(recent_sessions) >= limit:
                        break
            
            return recent_sessions
        except Exception as e:
            print(f"[MemoryManager] Error getting recent sessions: {e}")
            return []
    
    def get_sessions_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get sessions filtered by category.
        Useful for analyzing user patterns or generating category-specific insights.
        """
        try:
            # Filter sessions by category
            category_sessions = [
                session for session in self.memory 
                if session["category"] == category
            ]
            
            # Return most recent first (sorted by timestamp)
            return sorted(category_sessions, 
                         key=lambda x: x["timestamp"], 
                         reverse=True)[:limit]
        except Exception as e:
            print(f"[MemoryManager] Error getting category sessions: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memory.
        Useful for monitoring system health and memory usage.
        """
        try:
            if not self.memory:
                return {"total_sessions": 0, "categories": {}, "recent_activity": 0}
            
            category_counts = {}
            recent_count = 0
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for session in self.memory:
                # Count sessions by category for distribution analysis
                category = session["category"]
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count recent activity (last 7 days)
                session_date = datetime.fromisoformat(session["timestamp"])
                if session_date >= cutoff_date:
                    recent_count += 1
            
            return {
                "total_sessions": len(self.memory),
                "categories": category_counts,
                "recent_activity": recent_count,
                "memory_usage": f"{len(self.memory)}/{self.max_sessions}",
                "oldest_session": self.memory[0]["timestamp"] if self.memory else None,
                "newest_session": self.memory[-1]["timestamp"] if self.memory else None
            }
        except Exception as e:
            print(f"[MemoryManager] Error getting memory stats: {e}")
            return {"error": str(e)}
    
    def _load_memory(self) -> deque:
        """
        Load memory from disk with error handling.
        Creates backup if file is corrupted.
        """
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "rb") as f:
                    loaded_memory = pickle.load(f)
                    print(f"[MemoryManager] Loaded {len(loaded_memory)} sessions from {self.memory_file}")
                    return loaded_memory
        except Exception as e:
            print(f"[MemoryManager] Error loading memory file: {e}")
            # Try to backup corrupted file before overwriting
            if os.path.exists(self.memory_file):
                backup_file = self.memory_file + ".backup"
                try:
                    os.rename(self.memory_file, backup_file)
                    print(f"[MemoryManager] Backed up corrupted file to {backup_file}")
                except:
                    pass
        
        # Initialize fresh memory if load failed
        print(f"[MemoryManager] Initializing new memory with max {self.max_sessions} sessions")
        return deque(maxlen=self.max_sessions)
    
    def _save_memory(self):
        """
        Persist memory to disk with error handling.
        Creates backup before overwriting for safety.
        """
        try:
            # Create backup of existing file before saving new version
            if os.path.exists(self.memory_file):
                backup_file = self.memory_file + ".bak"
                with open(self.memory_file, "rb") as src, open(backup_file, "wb") as dst:
                    dst.write(src.read())
            
            # Save new memory state
            with open(self.memory_file, "wb") as f:
                pickle.dump(self.memory, f)
                
        except Exception as e:
            print(f"[MemoryManager] Error saving memory: {e}")
    
    def _generate_session_id(self, query: str, response: str) -> str:
        """
        Generate unique session ID using MD5 hash.
        Includes timestamp to ensure uniqueness even for identical conversations.
        """
        content = f"{query}_{response}_{datetime.now().isoformat()}"
        return "session_" + hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _categorize_session(self, text: str) -> str:
        """
        Categorize session based on content keywords.
        Uses keyword matching to assign sessions to predefined categories.
        """
        text_lower = text.lower()
        
        # Check each category's keywords in order of priority
        for category, keywords in self.session_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        # Default to general_advice if no keywords match
        return "general_advice"
    
    def _create_session_text(self, query: str, response: str, 
                           web_context: Optional[List[Dict]], 
                           data_insights: Optional[str]) -> str:
        """
        Create comprehensive text for embedding.
        Combines all session components into single text for holistic semantic representation.
        """
        # Start with query and response
        parts = [f"Query: {query}", f"Response: {response}"]
        
        # Add web context summaries if available
        if web_context:
            context_summaries = [item.get("summary", "") for item in web_context]
            if context_summaries:
                parts.append(f"Context: {' '.join(context_summaries)}")
        
        # Add data insights if available
        if data_insights:
            parts.append(f"Data: {data_insights}")
        
        # Join with separator for clear structure
        return " | ".join(parts)
    
    def _create_basic_summary(self, query: str, response: str) -> str:
        """
        Create basic summary when none provided.
        Truncates to reasonable lengths for display.
        """
        return f"Q: {query[:100]}... A: {response[:200]}..."
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        Returns value between -1 (opposite) and 1 (identical).
        Standard measure for semantic similarity in NLP.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _calculate_recency_score(self, timestamp: str, recency_bias: float) -> float:
        """
        Calculate recency score based on session age.
        Uses exponential decay with 30-day half-life for gradual degradation.
        """
        try:
            session_date = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - session_date).days
            
            # Exponential decay: more recent sessions get higher scores
            # 30-day half-life means score halves every 30 days
            recency_score = recency_bias * np.exp(-age_days / 30)
            return min(recency_score, recency_bias)  # Cap at recency_bias
        except:
            return 0.0
    
    def clear_memory(self, confirm: bool = False):
        """
        Clear all memory (requires confirmation).
        Safety mechanism to prevent accidental data loss.
        """
        if confirm:
            self.memory.clear()
            self._save_memory()
            print("[MemoryManager] Memory cleared")
        else:
            print("[MemoryManager] Memory clear requires confirmation=True")
    
    def export_memory(self, export_file: str, format_type: str = "json"):
        """
        Export memory to different formats.
        Converts numpy embeddings to lists for JSON serialization.
        """
        try:
            if format_type == "json":
                # Convert numpy arrays to lists for JSON serialization
                export_data = []
                for session in self.memory:
                    export_session = session.copy()
                    # Convert embeddings from numpy arrays to JSON-serializable lists
                    export_session["embeddings"] = {
                        key: emb.tolist() for key, emb in session["embeddings"].items()
                    }
                    export_data.append(export_session)
                
                with open(export_file, "w") as f:
                    json.dump(export_data, f, indent=2)
                    
            print(f"[MemoryManager] Exported {len(self.memory)} sessions to {export_file}")
        except Exception as e:
            print(f"[MemoryManager] Error exporting memory: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the memory manager.
        Returns configuration and feature details for system introspection.
        """
        return {
            "embedding_model": self.embedding_model_name,
            "memory_file": self.memory_file,
            "max_sessions": self.max_sessions,
            "current_sessions": len(self.memory),
            "session_categories": list(self.session_categories.keys()),
            "features": [
                "Semantic similarity search",
                "Category-based filtering", 
                "Recency bias",
                "Comprehensive session storage",
                "Persistent memory"
            ]
        }