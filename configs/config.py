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

    # ── Neo4j (Graph-RAG) ────────────────────────────────────────
    NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "abcd1234")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    @staticmethod
    def validate():
        missing = []
        if not Config.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not Config.MILVUS_URI:
            missing.append("MILVUS_URI")
        if not Config.NEO4J_URI:
            missing.append("NEO4J_URI")
        if missing:
            raise ValueError(f"Thiếu các biến môi trường: {', '.join(missing)}")
