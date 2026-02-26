from sentence_transformers import SentenceTransformer
from configs.config import Config

class EmbeddingManager:
    def __init__(self):
        # Load model một lần duy nhất khi khởi tạo
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)

    def encode(self, text):
        return self.model.encode(text)
