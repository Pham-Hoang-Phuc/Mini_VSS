"""
ingestion_pipeline.py
=====================
Ingestion Pipeline: nạp dữ liệu vào Milvus (Vector-RAG) và ArcadeDB (Graph-RAG).

Tách từ main.py run_ingestion().
"""

from src.knowledge.embedding import EmbeddingManager
from src.knowledge.entity_extractor import EntityExtractor
from src.storage.milvus_store import MilvusManager
from src.storage.arcadedb_store import ArcadeDBGraph
from src.retrieval.graph_search import GraphSearchManager


def run_ingestion(embedder: EmbeddingManager,
                  milvus_db: MilvusManager,
                  graph: ArcadeDBGraph,
                  graph_search: GraphSearchManager,
                  entity_extractor: EntityExtractor,
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
    print("\n[2/2] Graph-RAG → ArcadeDB")
    graph.clear_all()   # reset graph để tránh duplicate khi chạy lại

    total_rel = 0
    for i, chunk in enumerate(video_chunks_data):
        print(f"  [{i+1}/{len(video_chunks_data)}] {chunk['caption'][:55]}...")
        count = entity_extractor.extract_and_store(caption=chunk["caption"])
        total_rel += count

    print(f"\n  [ArcadeDB] ✓ Đã nạp {total_rel} relationships từ {len(video_chunks_data)} chunks.")

    # Khởi tạo chain sau khi graph đã có dữ liệu
    graph_search.setup_chain()

    print("\n" + "="*55)
    print("  ✓ Ingestion hoàn tất! Sẵn sàng nhận truy vấn.")
    print("="*55 + "\n")
