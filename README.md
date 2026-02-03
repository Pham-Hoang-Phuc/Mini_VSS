# Mini VSS - Hệ thống Tìm kiếm Ngữ nghĩa trong Video

Đây là một dự án demo xây dựng một hệ thống Tìm kiếm Ngữ nghĩa cho Video (Video Semantic Search - VSS) đơn giản. Hệ thống cho phép người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên để tìm kiếm các phân đoạn liên quan trong một video.

## Kiến trúc

Hệ thống bao gồm các thành phần chính:

1.  **Ứng dụng Python (`main.py`):**
    *   Sử dụng `sentence-transformers` để chuyển đổi cả nội dung video (chú thích) và câu hỏi của người dùng thành các vector embedding.
    *   Kết nối và tương tác với Milvus để lưu trữ và tìm kiếm các vector này.
    *   Sử dụng Google Gemini API (`gemini-2.5-flash-lite`) để tạo ra câu trả lời dựa trên kết quả tìm kiếm từ Milvus.

2.  **Cơ sở dữ liệu Vector (`Milvus`):**
    *   Chạy trong Docker cùng với các dịch vụ phụ thuộc là `etcd` và `minio`.
    *   Lưu trữ các vector embedding của các phân đoạn video.
    *   Thực hiện tìm kiếm tương đồng (similarity search) để tìm các vector gần nhất với vector câu hỏi.

3.  **Mô hình Ngôn ngữ Lớn (`Google Gemini`):**
    *   Nhận kết quả tìm kiếm (context) từ Milvus và câu hỏi gốc của người dùng.
    *   Tổng hợp thông tin và đưa ra câu trả lời mạch lạc, tự nhiên.

## Yêu cầu

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt:

*   [Docker](https://www.docker.com/get-started) và [Docker Compose](https://docs.docker.com/compose/install/)
*   [Python 3.10](https://www.python.org/downloads/) và `pip`
*   Một **API Key** từ Google AI Studio cho mô hình Gemini.

## Hướng dẫn Cài đặt & Chạy

### 1. Tải mã nguồn

```bash
git clone https://github.com/Pham-Hoang-Phuc/Mini_VSS.git
```

### 2. Cấu hình Biến môi trường

Tạo một file tên là `.env` ở thư mục gốc của dự án.

Sao chép nội dung dưới đây vào file `.env` của bạn:

```env
# Thay YOUR_GEMINI_API_KEY bằng API key của bạn
YOUR_GEMINI_API_KEY="AIzaSyXXXXXXXXXXXXXXXXXXX"

# URI để kết nối tới Milvus chạy trong Docker
MILVUS_URI="http://localhost:19530"
```

**Quan trọng:** Thay thế `AIzaSyXXXXXXXXXXXXXXXXXXX` bằng Google Gemini API Key thực của bạn.

### 3. Cài đặt các thư viện Python

Mở terminal và chạy lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 4. Khởi chạy các dịch vụ Docker

Sử dụng Docker Compose để khởi chạy Milvus và các dịch vụ liên quan.

```bash
docker-compose up -d
```

Lệnh này sẽ tải các image cần thiết và khởi chạy container ở chế độ nền (`-d`). Quá trình này có thể mất vài phút trong lần chạy đầu tiên.

Để kiểm tra các container đang chạy, bạn có thể dùng lệnh `docker ps`. Bạn sẽ thấy 3 container có tên `milvus-standalone`, `milvus-minio`, và `milvus-etcd`.

### 5. Chạy ứng dụng

Sau khi các container đã khởi động và ổn định (chờ khoảng 1-2 phút), bạn có thể chạy ứng dụng chính:

```bash
python main.py
```

Khi chạy lần đầu, ứng dụng sẽ:
1.  **Khởi tạo Collection:** Tạo cấu trúc dữ liệu trong Milvus.
2.  **Nạp dữ liệu:** Chuyển đổi các chú thích video trong `video_chunks.py` thành vector và nạp vào Milvus.

Sau đó, chương trình sẽ hiển thị một dòng chờ để bạn nhập câu hỏi:

```
Bạn muốn tìm gì trong video?:
```

Bạn có thể nhập câu hỏi của mình (ví dụ: "con mèo đang làm gì?") và nhấn Enter. AI sẽ phân tích và trả về kết quả.

Để thoát khỏi chương trình, nhập `exit` và nhấn Enter.

## Dừng các dịch vụ

Để tắt các container của Milvus, sử dụng lệnh:

```bash
docker-compose down
```

Lệnh này sẽ dừng và xóa các container nhưng vẫn giữ lại dữ liệu trong thư mục `volumes` (nếu bạn muốn xóa cả dữ liệu, dùng `docker-compose down -v`).
