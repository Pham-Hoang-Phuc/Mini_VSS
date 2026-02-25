"""
query.py
========
Chỉ chạy CA-RAG Query Loop — KHÔNG nạp dữ liệu vào database.

Yêu cầu: đã chạy main.py ít nhất 1 lần để Milvus và Neo4j có dữ liệu.
"""

from config import Config
from embedding import EmbeddingManager
from database import MilvusManager
from llm import GeminiManager
from graph_rag import GraphRAGManager
from main import run_query


def main():
    Config.validate()

    print("Đang khởi tạo CA-RAG (chỉ Query, không Ingestion)...")
    embedder  = EmbeddingManager()
    milvus_db = MilvusManager()
    graph_db  = GraphRAGManager()
    ai        = GeminiManager()

    # Tải collection Milvus (không tạo lại)
    milvus_db.client.load_collection(Config.COLLECTION_NAME)

    # Khởi tạo Cypher chain từ schema Neo4j hiện có
    graph_db.setup_chain()

    print("\nSẵn sàng! Nhập câu hỏi về video (gõ 'exit' để thoát).\n")
    while True:
        try:
            user_query = input("Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit", "thoát"):
            print("Tạm biệt!")
            break

        answer = run_query(user_query, embedder, milvus_db, graph_db, ai)
        print(f"\n[AI]: {answer}\n")


if __name__ == "__main__":
    main()
