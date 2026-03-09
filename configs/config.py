import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── Gemini / Google AI ──────────────────────────────────────
    GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
    GEMINI_BASE_URL  = os.getenv(
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # ── Milvus (Vector-RAG) ──────────────────────────────────────
    MILVUS_URI       = os.getenv("MILVUS_URI", "http://localhost:19530")
    COLLECTION_NAME  = "vss_demo"

    # ── Model tên ────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # SentenceTransformer
    LLM_MODEL_NAME       = "gemma-3-12b-it"  # Gemini SDK (gen answer)
    GRAPH_LLM_MODEL      = "gemma-3-27b-it"     # entity extraction (openai compat)
    CYPHER_LLM_MODEL     = "gemma-3-12b-it"     # sinh Cypher (langchain)
    QA_LLM_MODEL         = "gemma-3-12b-it"      # tổng hợp câu trả lời (langchain)

    # ── ArcadeDB (Graph-RAG) ─────────────────────────────────────
    ARCADEDB_HOST     = os.getenv("ARCADEDB_HOST",     "127.0.0.1")
    ARCADEDB_PORT     = os.getenv("ARCADEDB_PORT",     "2480")
    ARCADEDB_USERNAME = os.getenv("ARCADEDB_USERNAME", "root")
    ARCADEDB_PASSWORD = os.getenv("ARCADEDB_PASSWORD", "abcd1234")
    ARCADEDB_DATABASE = os.getenv("ARCADEDB_DATABASE", "VisionMetada")

    @staticmethod
    def validate():
        missing = []
        if not Config.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not Config.MILVUS_URI:
            missing.append("MILVUS_URI")
        if not Config.ARCADEDB_HOST:
            missing.append("ARCADEDB_HOST")
        if missing:
            raise ValueError(f"Thiếu các biến môi trường: {', '.join(missing)}")
