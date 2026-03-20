"""
minio_store.py
==============
Module lưu trữ Video Chunks lên MinIO (Object Storage).

Pipeline (theo vss_context.md - Mục 4):
  - Lưu video chunk sau khi gom frames
  - URL video (minio_url) được lưu kèm metadata trong Milvus
  - Dùng làm bằng chứng (evidence) khi trả lời câu hỏi

TODO:
  - Implement MinIOManager class
  - Upload video chunk → bucket
  - Generate presigned URL cho query results
"""


class MinIOManager:
    """
    Quản lý upload/download video chunks trên MinIO.

    Usage tương lai:
        minio = MinIOManager(endpoint="localhost:9000", bucket="vss-videos")
        url = minio.upload(video_bytes, object_name="cam01/chunk_001.mp4")
    """

    def __init__(self, endpoint: str = "localhost:9000",
                 access_key: str = "minioadmin",
                 secret_key: str = "minioadmin",
                 bucket: str = "vss-videos"):
        self.endpoint = endpoint
        self.bucket = bucket
        # TODO: self.client = Minio(endpoint, access_key, secret_key, secure=False)

    def upload(self, video_bytes: bytes, object_name: str) -> str:
        """
        Upload video chunk lên MinIO.

        Returns:
            URL hoặc object path trên MinIO.
        TODO: Implement upload logic.
        """
        raise NotImplementedError("MinIOManager.upload() chưa được implement.")

    def get_url(self, object_name: str) -> str:
        """
        Tạo presigned URL cho video chunk.
        TODO: Implement presigned URL generation.
        """
        raise NotImplementedError("MinIOManager.get_url() chưa được implement.")
