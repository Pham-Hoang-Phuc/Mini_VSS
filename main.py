"""
main.py
=======
CA-RAG Pipeline hoàn chỉnh: Vector-RAG (Milvus) + Graph-RAG (ArcadeDB)

Flow:
  Ingestion Pipeline (chạy 1 lần):
    1. Embed caption  → Milvus    (Vector-RAG)
    2. Extract entity → ArcadeDB  (Graph-RAG)

  Query Loop:
    1. LLM Route: phân tích câu hỏi → chọn Vector / Graph / cả hai
    2. Parallel Retrieval:
         - Vector-RAG: Milvus similarity search
         - Graph-RAG : ArcadeDB Cypher reasoning
    3. LLM Synthesize: tổng hợp câu trả lời cuối cùng
"""

from configs.config import Config
from src.knowledge.embedding import EmbeddingManager
from src.knowledge.entity_extractor import EntityExtractor
from src.storage.milvus_store import MilvusManager
from src.storage.arcadedb_store import ArcadeDBGraph
from src.models.llm import GeminiManager
from src.retrieval.vector_search import VectorSearchManager
from src.retrieval.graph_search import GraphSearchManager
from src.pipeline.ingestion_pipeline import run_ingestion
from src.pipeline.query_pipeline import run_query
from data.samples.video_chunks_sample import chunks


def main():
    # 0. Validate config
    Config.validate()

    # 1. Khởi tạo các module
    print("\nĐang khởi tạo hệ thống CA-RAG...")
    embedder   = EmbeddingManager()
    milvus_db  = MilvusManager()
    graph      = ArcadeDBGraph(
        host=Config.ARCADEDB_HOST,
        port=Config.ARCADEDB_PORT,
        username=Config.ARCADEDB_USERNAME,
        password=Config.ARCADEDB_PASSWORD,
        database=Config.ARCADEDB_DATABASE,
    )
    print("  [GraphRAG] ✓ Kết nối ArcadeDB thành công.")

    graph_search      = GraphSearchManager(graph)
    entity_extractor  = EntityExtractor(graph)
    vector_search     = VectorSearchManager(embedder, milvus_db)
    ai                = GeminiManager()

    # 2. Ingestion Pipeline
    video_chunks_data = chunks.get_chunks()
    run_ingestion(embedder, milvus_db, graph, graph_search, entity_extractor, video_chunks_data)

    # 3. Query Loop
    print("Nhập câu hỏi về video (gõ 'exit' để thoát).\n")
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

        answer = run_query(user_query, vector_search, graph_search, ai)
        print(f"\n[AI]: {answer}\n")


if __name__ == "__main__":
    main()
