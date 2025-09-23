from sentence_transformers import SentenceTransformer

class EmbeddingAgent:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=True)
