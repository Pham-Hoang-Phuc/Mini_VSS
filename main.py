from config import Config
from embedding import EmbeddingManager
from database import MilvusManager
from llm import GeminiManager
from video_chunks import chunks

def main():
    # 0. Kiểm tra cấu hình
    Config.validate()

    # 1. Khởi tạo các module
    embedder = EmbeddingManager()
    db = MilvusManager()
    ai_assistant = GeminiManager()

    # 2. Setup Database & Ingestion (Chỉ làm lần đầu hoặc khi reset)
    print("--- Đang khởi tạo Database ---")
    db.setup_collection()

    video_chunks_data = chunks.get_chunks()

    print("--- Đang nạp dữ liệu vào Milvus ---")
    formatted_data = []
    for chunk in video_chunks_data:
        formatted_data.append({
            "video_vector": embedder.encode(chunk["caption"]),
            "caption": chunk["caption"],
            "timestamp": chunk["timestamp"]
        })
    db.insert_data(formatted_data)

    # 3. Luồng truy vấn
    while True:
        user_query = input("\nBạn muốn tìm gì trong video?: ")

        if user_query.lower() == "exit":
            break

        # Tìm kiếm Vector
        query_vector = embedder.encode(user_query)
        search_results = db.search(query_vector)

        # Tổng hợp context
        context_list = []
        for hits in search_results:
            for hit in hits:
                context_list.append(f"- Lúc {hit['entity']['timestamp']}: {hit['entity']['caption']}")
        
        context_text = "\n".join(context_list)

        # 4. LLM Trả lời
        print("\n--- AI Đang phân tích nội dung ---")
        answer = ai_assistant.generate_answer(user_query, context_text)
        
        print(f"\n[AI]: {answer}")

if __name__ == "__main__":
    main()