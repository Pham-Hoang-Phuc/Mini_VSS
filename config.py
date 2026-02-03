import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("YOUR_GEMINI_API_KEY")
    MILVUS_URI = os.getenv("MILVUS_URI")
    COLLECTION_NAME = "vss_demo"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    LLM_MODEL_NAME = "gemini-2.5-flash-lite"

    @staticmethod
    def validate():
        if not Config.GEMINI_API_KEY or not Config.MILVUS_URI:
            raise ValueError("Thiếu GEMINI_API_KEY hoặc MILVUS_URI trong file .env")