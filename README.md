# Mini VSS — CA-RAG: Context-Aware Retrieval-Augmented Generation

Hệ thống **CA-RAG** (Context-Aware RAG) dành cho phân tích nội dung video nhà máy thông minh.  
Kết hợp **Vector-RAG** (tìm kiếm ngữ nghĩa qua Milvus) và **Graph-RAG** (suy luận quan hệ qua Neo4j), điều phối tự động bởi Gemini LLM.

---

## Kiến trúc tổng quan

```
Video captions (VLM output)
         │
         ▼
┌────────────────────────┐
│   Ingestion Pipeline   │
│                        │
│  ┌─────────────────┐   │        ┌─────────────────────────┐
│  │  Vector-RAG     │───┼───────▶│  Milvus (Vector DB)     │
│  │  (embedding)    │   │        │  • caption (embedded)   │
│  └─────────────────┘   │        │  • timestamp            │
│                        │        │  • video_id             │
│  ┌─────────────────┐   │        │  • camera_id            │
│  │  Graph-RAG      │───┼───────▶│─────────────────────────│
│  │  (LLM extract)  │   │        │  Neo4j (Knowledge Graph)│
│  └─────────────────┘   │        │  • Entity nodes         │
└────────────────────────┘        │  • Relationships        │
                                  └─────────────────────────┘
         │
         ▼ (sau ingestion)
┌────────────────────────────────────────────────────────────┐
│                    CA-RAG Query Loop                       │
│                                                            │
│  User Query                                                │
│      │                                                     │
│      ▼                                                     │
│  [1] Router LLM (Gemini)                                   │
│      ├── use_vector? → Milvus similarity search            │
│      └── use_graph?  → Neo4j Cypher reasoning              │
│                                                            │
│  [2] Parallel Retrieval                                    │
│      ├── Vector-RAG: top-K caption theo ngữ nghĩa          │
│      └── Graph-RAG:  entity + relationship từ đồ thị       │
│                                                            │
│  [3] Synthesize LLM (Gemini)                               │
│      └── Câu trả lời cuối cùng                             │
└────────────────────────────────────────────────────────────┘
```

---

## Cấu trúc project

```
mini_vss/
├── main.py            # Entry point: chạy ingestion + query loop
├── query.py           # Chỉ chạy query loop (khi đã có data sẵn)
├── config.py          # Cấu hình toàn hệ thống (load từ .env)
│
├── video_chunks.py    # Dữ liệu mẫu: caption, timestamp, video_id, camera_id
├── embedding.py       # EmbeddingManager — SentenceTransformer
├── database.py        # MilvusManager   — kết nối & thao tác Vector DB
├── graph_rag.py       # GraphRAGManager — kết nối Neo4j, extract entity/relation, query
├── llm.py             # GeminiManager   — router, generate answer
│
├── docker-compose.yml # Khởi động Milvus (etcd + minio + standalone)
├── .env               # Biến môi trường (API key, DB connection)
├── .env.example       # Template .env
└── requirements.txt   # Python dependencies
```

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Python | ≥ 3.10 |
| Docker & Docker Compose | ≥ 24 |
| Neo4j | ≥ 5.x (Community OK) |
| Gemini API Key | [Google AI Studio](https://aistudio.google.com/) |

---

## Hướng dẫn cài đặt

### 1. Clone & tạo môi trường

```bash
git clone <repo-url>
cd mini_vss

conda create -n mini_vss python=3.10 -y
conda activate mini_vss

pip install -r requirements.txt
```

### 2. Cấu hình `.env`

```bash
cp .env.example .env
```

Chỉnh sửa file `.env`:

```env
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

NEO4J_URI="neo4j://[IP_ADDRESS]"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_password"
NEO4J_DATABASE="neo4j"
```

> **Lưu ý:** Neo4j Community Edition chỉ hỗ trợ database tên `neo4j`. Không đổi thành tên khác.

### 3. Khởi động Milvus (Docker)

```bash
docker compose up -d
```

Đợi khoảng 30–60 giây để Milvus khởi động xong. Kiểm tra:

```bash
docker compose ps        # Đảm bảo 3 service đang healthy
```

### 4. Khởi động Neo4j

Dùng [Neo4j Desktop](https://neo4j.com/download/) hoặc Docker:

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5
```

---

## ▶️ Cách sử dụng

### Lần đầu: Ingestion + Query

```bash
python main.py
```

Pipeline sẽ:
1. Embed caption → nạp vào **Milvus** (kèm `timestamp`, `video_id`, `camera_id`)
2. Gọi Gemini trích xuất entity/relation từ caption → nạp vào **Neo4j** (knowledge graph thuần túy)
3. Khởi động query loop

### Từ lần sau: Chỉ Query (không nạp lại data)

```bash
python query.py
```

Dùng khi Milvus và Neo4j đã có dữ liệu sẵn từ lần chạy trước.

### Ví dụ câu hỏi

```
Câu hỏi: Xe nâng làm gì trong video?
Câu hỏi: Tìm cảnh có công nhân kiểm tra chất lượng
Câu hỏi: Đội bảo trì bảo trì cái gì?
Câu hỏi: Lúc nào có người ở khu vực đóng gói?
```

---

## Chi tiết kỹ thuật

### Phân chia lưu trữ

| | Milvus (Vector DB) | Neo4j (Graph DB) |
|---|---|---|
| **Lưu gì** | caption embedding + metadata | Entity nodes + Relationships |
| **Fields** | `caption`, `timestamp`, `video_id`, `camera_id` | `(Entity)-[RELATIONSHIP]->(Entity)` |
| **Dùng cho** | Tìm kiếm ngữ nghĩa (semantic) | Suy luận quan hệ (reasoning) |

### Models sử dụng

| Mục đích | Model |
|---|---|
| Embedding (local) | `all-MiniLM-L6-v2` (SentenceTransformer) |
| Entity extraction | `gemma-3-27b-it` (Gemini API) |
| Cypher generation | `gemma-3-12b-it` (LangChain) |
| QA & Router | `gemma-3-12b-it` (Gemini SDK) |

### Router logic

Gemini tự động phân tích câu hỏi và chọn công cụ:

| Loại câu hỏi | Công cụ |
|---|---|
| "Tìm cảnh...", "Lúc nào có..." | Vector-RAG (Milvus) |
| "Ai làm gì?", "Quan hệ giữa..." | Graph-RAG (Neo4j) |
| Câu hỏi phức hợp | Cả hai |

## Lưu ý

- Dữ liệu mẫu trong `video_chunks.py` là caption giả lập từ camera nhà máy. Trong production, thay bằng output thật từ VLM (Vision-Language Model).
- Mỗi lần chạy `main.py`, collection Milvus và graph Neo4j đều bị **reset và nạp lại** để tránh duplicate.
- Dùng `query.py` nếu chỉ muốn hỏi mà không nạp dữ liệu lại.
