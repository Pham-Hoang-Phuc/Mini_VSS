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
from src.engine.embedding import EmbeddingManager
from src.database.database import MilvusManager
from src.models.llm import GeminiManager
from src.engine.graph_rag import GraphRAGManager
from src.utils.video_chunks import chunks


# ──────────────────────────────────────────────────────────────
# Ingestion Pipeline
# ──────────────────────────────────────────────────────────────
def run_ingestion(embedder: EmbeddingManager,
                  milvus_db: MilvusManager,
                  graph_db: GraphRAGManager,
                  video_chunks_data: list):
    """
    Nạp dữ liệu vào cả Milvus và ArcadeDB.
    Gọi 1 lần khi khởi động (hoặc khi reset dữ liệu).
    """
    print("\n" + "="*55)
    print("  INGESTION PIPELINE")
    print("="*55)

    # ── Vector-RAG: đẩy embedding vào Milvus (lưu đủ metadata) ──
    print("\n[1/2] Vector-RAG → Milvus")
    milvus_db.setup_collection()

    formatted_data = []
    for chunk in video_chunks_data:
        formatted_data.append({
            "video_vector": embedder.encode(chunk["caption"]).tolist(),
            "caption":      chunk["caption"],
            "timestamp":    chunk["timestamp"],
            "video_id":     chunk.get("video_id", ""),
            "camera_id":    chunk.get("camera_id", ""),
        })
    milvus_db.insert_data(formatted_data)
    print(f"  [Milvus] ✓ Đã nạp {len(formatted_data)} chunks.")

    # ── Graph-RAG: trích xuất entity và đẩy vào ArcadeDB ──────
    # ArcadeDB chỉ lưu caption (knowledge graph thuần tuý, không metadata)
    print("\n[2/2] Graph-RAG → ArcadeDB")
    graph_db.clear_graph()   # reset graph để tránh duplicate khi chạy lại

    total_rel = 0
    for i, chunk in enumerate(video_chunks_data):
        print(f"  [{i+1}/{len(video_chunks_data)}] {chunk['caption'][:55]}...")
        count = graph_db.extract_and_store(caption=chunk["caption"])
        total_rel += count

    print(f"\n  [ArcadeDB] ✓ Đã nạp {total_rel} relationships từ {len(video_chunks_data)} chunks.")

    # Khởi tạo chain sau khi graph đã có dữ liệu
    graph_db.setup_chain()

    print("\n" + "="*55)
    print("  ✓ Ingestion hoàn tất! Sẵn sàng nhận truy vấn.")
    print("="*55 + "\n")


# ──────────────────────────────────────────────────────────────
# CA-RAG Query
# ──────────────────────────────────────────────────────────────
def run_query(query: str,
              embedder: EmbeddingManager,
              milvus_db: MilvusManager,
              graph_db: GraphRAGManager,
              ai: GeminiManager) -> str:
    """
    CA-RAG query pipeline:
      1. Route  → chọn Vector / Graph / cả hai
      2. Retrieve song song
      3. Synthesize → câu trả lời cuối
    """
    print(f"\n{'─'*55}")
    print(f"  Câu hỏi: {query}")
    print(f"{'─'*55}")

    # ── Bước 1: Query Parsing / Routing ───────────────────────
    route = ai.route_query(query)
    use_vector = route.get("use_vector", True)
    use_graph  = route.get("use_graph",  True)
    reason     = route.get("reason", "")

    tools_used = []
    if use_vector: tools_used.append("Vector-RAG")
    if use_graph:  tools_used.append("Graph-RAG")
    print(f"  [Router] Công cụ: {' + '.join(tools_used) or 'Không xác định'}")
    if reason:
        print(f"  [Router] Lý do: {reason}")

    # ── Bước 2: Parallel Retrieval ────────────────────────────
    vector_context = ""
    graph_context  = ""

    if use_vector:
        print("  → [Vector-RAG] Đang tìm kiếm Milvus...")
        query_vector = embedder.encode(query)
        results = milvus_db.search(query_vector, limit=3)
        ctx_lines = []
        for hits in results:
            for hit in hits:
                e = hit['entity']
                ctx_lines.append(
                    f"  - [{e['timestamp']}] [{e.get('video_id','')} / {e.get('camera_id','')}] {e['caption']}"
                )
        vector_context = "\n".join(ctx_lines)
        if vector_context:
            print(f"  → [Vector-RAG] Tìm thấy {len(ctx_lines)} kết quả.")
        else:
            print("  → [Vector-RAG] Không có kết quả.")

    if use_graph:
        print("  → [Graph-RAG] Đang truy vấn ArcadeDB...")
        graph_context = graph_db.query(query)
        if graph_context:
            print("  → [Graph-RAG] Có kết quả từ đồ thị.")
        else:
            print("  → [Graph-RAG] Không có kết quả từ đồ thị.")

    # ── Bước 3: Synthesize ────────────────────────────────────
    print("  → [LLM] Đang tổng hợp câu trả lời...")
    if use_vector and use_graph:
        answer = ai.generate_answer_combined(query, vector_context, graph_context)
    elif use_vector:
        answer = ai.generate_answer(query, vector_context)
    elif use_graph:
        answer = graph_context if graph_context else "Graph-RAG không tìm thấy kết quả."
    else:
        answer = "Không xác định được công cụ phù hợp."

    return answer


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # 0. Validate config
    Config.validate()

    # 1. Khởi tạo các module
    print("\nĐang khởi tạo hệ thống CA-RAG...")
    embedder   = EmbeddingManager()
    milvus_db  = MilvusManager()
    graph_db   = GraphRAGManager()
    ai         = GeminiManager()

    # 2. Ingestion Pipeline
    video_chunks_data = chunks.get_chunks()
    run_ingestion(embedder, milvus_db, graph_db, video_chunks_data)

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

        answer = run_query(user_query, embedder, milvus_db, graph_db, ai)
        print(f"\n[AI]: {answer}\n")


if __name__ == "__main__":
    main()
