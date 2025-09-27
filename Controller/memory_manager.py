import os
import pickle
import json
from collections import deque
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import re


class MemoryManager:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", 
                 memory_file="financial_memory.pkl", max_sessions=200):
        """
        Initialize Memory Manager for financial advisory sessions.
        
        Args:
            embedding_model_name: SentenceTransformer model for embeddings
            memory_file: File path for persistent storage
            max_sessions: Maximum number of sessions to store
        """
        self.embedding_model_name = embedding_model_name
        self.memory_file = memory_file
        self.max_sessions = max_sessions
        self.verified_companies_cache = {}
        self.cache = {}

        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"[MemoryManager] Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            print(f"[MemoryManager] Error loading embedding model: {e}")
            raise
        
        # Load existing memory or initialize empty
        self.memory = self._load_memory()
        
        # Session categories for better organization
        self.session_categories = {
            "portfolio_analysis": ["portfolio", "diversification", "allocation", "holdings"],
            "stock_analysis": ["stock", "share", "equity", "company", "JSE"],
            "market_insight": ["market", "economy", "sector", "trend", "outlook"],
            "risk_management": ["risk", "volatility", "hedge", "protection", "insurance"],
            "general_advice": ["invest", "advice", "recommend", "strategy", "plan"]
        }
    
    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value):
        self.cache[key] = value

    def get_verified_companies(self, exchange: str) -> dict:
        """
        Retrieve verified companies for a given exchange.
        Returns empty dict if none cached.
        """
        return self.verified_companies_cache.get(exchange, {})

    def set_verified_companies(self, exchange: str, companies: dict):
        """
        Store verified companies for a given exchange.
        """
        self.verified_companies_cache[exchange] = companies
    def _sanitize_text(self, text: str) -> str:
        """
        Clean advisor responses before saving to memory.
        Removes internal markers, instructions, and noise.
        """
        if not text:
            return text

        # Remove [INTERNAL ...], [NOTE ...], [REVISION ...]
        text = re.sub(r"\[.*?(INTERNAL|NOTE|REVISION).*?\]", "", text,
                      flags=re.IGNORECASE | re.DOTALL)

        # Remove stray END statements (e.g., END OF ANALYSIS, END RESPONSE, etc.)
        text = re.sub(r"\bEND[^\n]*", "", text, flags=re.IGNORECASE)

        # Remove ALL CAPS leakage blocks like "THIS IS INTERNAL..."
        text = re.sub(r"[A-Z\s]{8,}", "", text)

        # Clean whitespace
        return text.strip()

    def add_session(self, query: str, advisor_response: str, web_context: Optional[List[Dict]] = None,
               data_insights: Optional[str] = None, summary: Optional[str] = None,
               user_metadata: Optional[Dict] = None) -> str:
        try:
            # Sanitize advisor response
            advisor_response = self._sanitize_text(advisor_response)

            # Create session ID
            session_id = self._generate_session_id(query, advisor_response)

            # Categorize session
            session_category = self._categorize_session(query + " " + advisor_response)

            # Create comprehensive session text for embedding
            full_session_text = self._create_session_text(
                query, advisor_response, web_context, data_insights
            )

            # Sanitize full session text too
            full_session_text = self._sanitize_text(full_session_text)

            # Generate embeddings
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            response_embedding = self.embedding_model.encode(advisor_response, convert_to_tensor=False)
            full_embedding = self.embedding_model.encode(full_session_text, convert_to_tensor=False)

            # Session data structure
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
                    **(user_metadata or {})
                }
            }

            # Add to memory
            self.memory.append(session_data)
            self._save_memory()

            print(f"[MemoryManager] Added session {session_id} (category: {session_category})")
            return session_id

        except Exception as e:
            print(f"[MemoryManager] Error adding session: {e}")
            return "error_session_" + str(hash(query))[:8]

    
    def search_memory(self, query_text: str, top_k: int = 3, 
                 category_filter: Optional[str] = None,
                 recency_bias: float = 0.1,
                     similarity_threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        Search memory for relevant past sessions with advanced filtering.
    
        Args:
            query_text: Current query to search for similar sessions
            top_k: Number of top similar sessions to return
            category_filter: Force filter by session category (optional)
            recency_bias: Weight given to recent sessions (0-1)
            similarity_threshold: Minimum similarity score to include
    
        Returns:
            List of relevant session data
        """
        try:
            if not self.memory:
                return []

            query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=False)
            query_category = self._categorize_session(query_text)

            # If no explicit filter, force category alignment
            active_category = category_filter or query_category

            scored_sessions = []
            for session in self.memory:
                # Skip if category doesn't match
                if session["category"] != active_category:
                    continue

                # Calculate similarities
                full_similarity = self._cosine_similarity(
                    query_embedding, session["embeddings"]["full_session"]
                )
                query_similarity = self._cosine_similarity(
                    query_embedding, session["embeddings"]["query"]
                )

                base_score = 0.7 * query_similarity + 0.3 * full_similarity
                recency_score = self._calculate_recency_score(session["timestamp"], recency_bias)

                final_score = base_score + recency_score

                if final_score >= similarity_threshold:
                    scored_sessions.append((final_score, session))

            scored_sessions.sort(key=lambda x: x[0], reverse=True)

            results = []
            for score, session in scored_sessions[:top_k]:
                result_session = session.copy()
                result_session["relevance_score"] = float(score)
                results.append(result_session)

            print(f"[MemoryManager] Found {len(results)} relevant sessions in category '{active_category}'")
            return results

        except Exception as e:
            print(f"[MemoryManager] Error searching memory: {e}")
            return []

    
    def get_recent_sessions(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions within specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_sessions = []
            
            for session in reversed(self.memory):  # Most recent first
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
        """Get sessions filtered by category"""
        try:
            category_sessions = [
                session for session in self.memory 
                if session["category"] == category
            ]
            
            # Return most recent first
            return sorted(category_sessions, 
                         key=lambda x: x["timestamp"], 
                         reverse=True)[:limit]
        except Exception as e:
            print(f"[MemoryManager] Error getting category sessions: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memory"""
        try:
            if not self.memory:
                return {"total_sessions": 0, "categories": {}, "recent_activity": 0}
            
            category_counts = {}
            recent_count = 0
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for session in self.memory:
                # Count by category
                category = session["category"]
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count recent activity
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
        """Load memory from disk with error handling"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "rb") as f:
                    loaded_memory = pickle.load(f)
                    print(f"[MemoryManager] Loaded {len(loaded_memory)} sessions from {self.memory_file}")
                    return loaded_memory
        except Exception as e:
            print(f"[MemoryManager] Error loading memory file: {e}")
            # Try to backup corrupted file
            if os.path.exists(self.memory_file):
                backup_file = self.memory_file + ".backup"
                try:
                    os.rename(self.memory_file, backup_file)
                    print(f"[MemoryManager] Backed up corrupted file to {backup_file}")
                except:
                    pass
        
        print(f"[MemoryManager] Initializing new memory with max {self.max_sessions} sessions")
        return deque(maxlen=self.max_sessions)
    
    def _save_memory(self):
        """Persist memory to disk with error handling"""
        try:
            # Create backup of existing file
            if os.path.exists(self.memory_file):
                backup_file = self.memory_file + ".bak"
                with open(self.memory_file, "rb") as src, open(backup_file, "wb") as dst:
                    dst.write(src.read())
            
            # Save new memory
            with open(self.memory_file, "wb") as f:
                pickle.dump(self.memory, f)
                
        except Exception as e:
            print(f"[MemoryManager] Error saving memory: {e}")
    
    def _generate_session_id(self, query: str, response: str) -> str:
        """Generate unique session ID"""
        content = f"{query}_{response}_{datetime.now().isoformat()}"
        return "session_" + hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _categorize_session(self, text: str) -> str:
        """Categorize session based on content keywords"""
        text_lower = text.lower()
        
        for category, keywords in self.session_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "general_advice"
    
    def _create_session_text(self, query: str, response: str, 
                           web_context: Optional[List[Dict]], 
                           data_insights: Optional[str]) -> str:
        """Create comprehensive text for embedding"""
        parts = [f"Query: {query}", f"Response: {response}"]
        
        if web_context:
            context_summaries = [item.get("summary", "") for item in web_context]
            if context_summaries:
                parts.append(f"Context: {' '.join(context_summaries)}")
        
        if data_insights:
            parts.append(f"Data: {data_insights}")
        
        return " | ".join(parts)
    
    def _create_basic_summary(self, query: str, response: str) -> str:
        """Create basic summary when none provided"""
        return f"Q: {query[:100]}... A: {response[:200]}..."
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _calculate_recency_score(self, timestamp: str, recency_bias: float) -> float:
        """Calculate recency score based on session age"""
        try:
            session_date = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - session_date).days
            
            # Exponential decay: more recent = higher score
            recency_score = recency_bias * np.exp(-age_days / 30)  # 30-day half-life
            return min(recency_score, recency_bias)  # Cap at recency_bias
        except:
            return 0.0
    
    def clear_memory(self, confirm: bool = False):
        """Clear all memory (requires confirmation)"""
        if confirm:
            self.memory.clear()
            self._save_memory()
            print("[MemoryManager] Memory cleared")
        else:
            print("[MemoryManager] Memory clear requires confirmation=True")
    
    def export_memory(self, export_file: str, format_type: str = "json"):
        """Export memory to different formats"""
        try:
            if format_type == "json":
                # Convert numpy arrays to lists for JSON serialization
                export_data = []
                for session in self.memory:
                    export_session = session.copy()
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
        """Get information about the memory manager"""
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