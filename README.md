# Mini VSS -- CA-RAG: Context-Aware Retrieval-Augmented Generation

Hệ thống CA-RAG (Context-Aware RAG) được thiết kế cho hệ thống phân tích nội dung video nhà máy thông minh.
Dự án kết hợp **Vector-RAG** (tìm kiếm ngữ nghĩa qua Milvus) và **Graph-RAG** (suy luận quan hệ qua ArcadeDB), được điều phối tự động bởi Gemini LLM, tạo thành một đường ống xử lý thông tin từ video theo thời gian thực.

## 1. Kiến trúc tổng quan hệ thống (System Pipeline)

Hệ thống hoạt động theo một đường ống xử lý (pipeline) từ trái sang phải như sau:
[Camera] -> [Redis] -> [Buffer] -> [VLM] -> [Databases] -> [Query] -> [User]

Chi tiết từng giai đoạn (Stage):

1. **Stage 1 - Ingestion (Nhận dữ liệu)**: Đọc frame ảnh và metadata từ các Edge Box tại camera, truyền qua Redis Streams.
2. **Stage 2 - Processing (Xử lý AI)**: Sử dụng cơ chế Sliding Window gom nhiều frames thành đoạn ngắn (Video Chunk). Đưa video chunk đó qua mô hình thị giác VLM (Qwen2-VL) để sinh ra mô tả hành động (Structured Caption) dưới dạng text.
3. **Stage 3 - Storage (Lưu trữ ban đầu)**: Video file được lưu trên MinIO (Object Storage) làm bằng chứng trực quan.
4. **Stage 4 - Knowledge Extraction (Trích xuất tri thức)**: 
   - Từ caption sinh ra ở bước trên, mô hình Embedding biến text thành các vector đại diện, lưu vào **Milvus** để tìm kiếm ngữ nghĩa (Vector-RAG).
   - LLM đọc caption để tách thành danh sách thực thể (entity) quy chuẩn và các quan hệ (relationships), đồng thời lưu vào **ArcadeDB** dưới dạng mạng lưới đồ thị (Graph-RAG).
5. **Stage 5 - Retrieval (Truy vấn mở rộng)**: Khi người dùng đặt câu hỏi:
   - LLM Router đoán ý định: Câu hỏi này nên tìm Vector (kiến thức hình ảnh tổng quát do VLM hiểu), hay tìm Graph (nhận diện logic quan hệ, hoạt động)? Hay là cả hai.
   - Hệ thống truy xuất song song cả Milvus lẫn ArcadeDB.
   - LLM Synthesis sẽ tổng hợp kết quả từ cả 2 nguồn và biên dịch thành câu trả lời cuối cùng để hiển thị cho người dùng (kèm theo timestamp hoặc video url từ MinIO).

> Hiện tại dự án đang ở giai đoạn Tích hợp Database và Query Loop (Stage 4 & 5). Các kiến trúc đọc Frame (Stage 1), xử lý Video (Stage 2), và lưu Object (Stage 3) đang được mock để xây dựng placeholder sẵn sàng cho việc tích hợp sắp tới.

---

## 2. Cấu trúc thư mục (Directory Structure)

```text
mini_vss/
├── main.py                - Điểm bắt đầu chương trình (Ingestion & Query Loop).
├── .env                   - Biến môi trường (API Keys, DB Configs).
├── configs/
│   └── config.py          - Quản lý cấu hình hệ thống.
├── data/
│   └── samples/
│       └── video_chunks_sample.py  - Dữ liệu mock output từ VLM.
├── deployments/
│   └── docker-compose.yml - Setup Milvus, etcd, MinIO, ArcadeDB.
└── src/
    ├── ingestion/         - Stage 1: Nhập dữ liệu luồng từ Stream server.
    │   └── redis_consumer.py
    ├── processing/        - Stage 2: Ghép Video & VLM Inference.
    │   ├── video_synthesizer.py
    │   └── vlm_reasoner.py
    ├── storage/           - Stage 3: Tầng kết nối (Drivers) tới CSDL.
    │   ├── milvus_store.py
    │   ├── arcadedb_store.py
    │   └── minio_store.py
    ├── knowledge/         - Stage 4: Biến đổi text thành tri thức (Worker).
    │   ├── embedding.py
    │   └── entity_extractor.py
    ├── retrieval/         - Stage 5: Tìm kiếm Hybrid RAG.
    │   ├── vector_search.py
    │   ├── graph_search.py
    │   └── hybrid_retriever.py
    ├── pipeline/          - Điều phối luồng thực thi (Orchestrator).
    │   ├── ingestion_pipeline.py
    │   └── query_pipeline.py
    └── models/            - Kết nối tới Model Cloud (Gemini SDK).
        └── llm.py
```

---

## 3. Mô hình hệ thống & Yêu cầu

Python version >= 3.10
Docker/Docker Compose version >= 24

| Điểm ứng dụng (Role)    | Tên Model được sử dụng                      |  
|-------------------------|---------------------------------------------|
| Embedding (Local)       | `all-MiniLM-L6-v2` (SentenceTransformer)    |
| Graph Entity Extraction | `gemma-3-27b-it` (Hỗ trợ LLM phân tích)     |
| Cypher SQL Generation   | `gemma-3-12b-it` (LangChain Graph tool)     |
| QA & Router logic       | `gemma-3-12b-it` (Gemini SDK)               |

---

## 4. Hướng dẫn cài đặt và thiết lập ban đầu

1. **Khởi tạo môi trường conda python**:
   ```bash
   conda create -n mini_vss python=3.10 -y
   conda activate mini_vss
   pip install -r requirements.txt
   ```

2. **Thiết lập chung (Environment Vars)**:
   ```bash
   cp .env.example .env
   # -> Thay đổi thông tin GEMINI_API_KEY bên trong .env hoặc thông số DB
   ```

3. **Khởi tạo các DB Containers (Docker)**:
   ```bash
   cd deployments/
   docker-compose up -d
   # Kiểm tra trạng thái: docker-compose ps
   # Lưu ý: Chờ khoảng vài chục giây để hệ thống DB (ArcadeDB, Milvus) khởi động ổn định.
   ```

4. **Chạy luồng chính & Trải nghiệm**:
   ```bash
   python main.py
   ```
   *Hệ thống chạy Ingestion đưa dữ liệu mẫu vào Vector/Graph DB.*
   *Sau đó, hệ thống sẽ chuyển sang vòng lặp Input CLI để người dùng đặt câu hỏi.*

---

## 5. Ví dụ truy vấn Demo

**Tra cứu theo hình ảnh (Vector DB):**
- *"Tìm cảnh có công nhân kiểm tra chất lượng sản phẩm bằng kính hiển vi"*
- *"Tìm đoạn nhà máy ban đêm lúc ca đêm bàn giao ở hành lang"*

**Tra cứu theo logic hành động (Graph DB):**
- *"Đội bảo trì đang bảo trì cái gì?"* (Trả về: đội bảo trì -> [kiểm_tra_định_kỳ] -> đường_ống_khí_nén)
- *"Ai kiểm tra các tờ thông số của bảng mạch?"*
- *"Xe nâng vận chuyển sản phẩm từ kho nào?"*

---

## Lưu ý 
- Dữ liệu mẫu (Caption giả từ camera nhà máy) hiện tại là Mock data trong file `data/samples/`. Ở giai đoạn sau sẽ được thay thế bằng kết quả thực tế từ `src/processing/vlm_reasoner`.
- Trong quá trình phát triển, mỗi lần chạy script `main.py`, dữ liệu cũ trên Milvus và ArcadeDB sẽ bị xóa và nạp lại từ đầu để đảm bảo môi trường sạch.

---
