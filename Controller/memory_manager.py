import os
import pickle
from collections import deque
from sentence_transformers import SentenceTransformer
import numpy as np

class MemoryManager:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", memory_file="memory.pkl", max_sessions=100):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.memory_file = memory_file
        self.max_sessions = max_sessions

        # Load existing memory or initialize empty
        if os.path.exists(memory_file):
            with open(memory_file, "rb") as f:
                self.memory = pickle.load(f)
        else:
            self.memory = deque(maxlen=max_sessions)  # each session: dict with 'summary', 'embedding', 'raw_text'

    def add_session(self, session_text):
        """Add a new session: summarize, embed, store"""
        embedding = self.embedding_model.encode(session_text, convert_to_tensor=False)
        session_data = {
            "summary": session_text,
            "embedding": embedding,
            "raw_text": session_text
        }
        self.memory.append(session_data)
        self._save_memory()

    def search_memory(self, query_text, top_k=3):
        """Return top_k similar past sessions"""
        if not self.memory:
            return []

        query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=False)
        similarities = []

        for session in self.memory:
            emb = session["embedding"]
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append(sim)

        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.memory[i] for i in top_indices]

    def _save_memory(self):
        """Persist memory to disk"""
        with open(self.memory_file, "wb") as f:
            pickle.dump(self.memory, f)

