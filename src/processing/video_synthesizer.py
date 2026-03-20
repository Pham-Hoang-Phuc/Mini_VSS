"""
video_synthesizer.py
====================
Module gom nhóm frames thành Video Chunks (Sliding Window).

Pipeline (theo vss_context.md - Mục 3, Bước 1):
  - Buffer N frames (50-100) thành 1 Video Chunk
  - Sử dụng bbox + obj_id từ Metadata để overlay thông tin lên video
  - Output: Video Chunk (.mp4) sẵn sàng cho VLM reasoning

TODO:
  - Implement sliding window buffer
  - Overlay bbox + obj_id lên frames (OpenCV)
  - Encode frames → video chunk (ffmpeg / OpenCV VideoWriter)
  - Upload video chunk lên MinIO
"""


class VideoSynthesizer:
    """
    Sliding Window Buffer: gom N frames → 1 Video Chunk.

    Usage tương lai:
        synthesizer = VideoSynthesizer(window_size=50, overlap=10)
        synthesizer.add_frame(frame_bytes, metadata)
        if synthesizer.is_ready():
            video_chunk = synthesizer.flush()
            vlm_reasoner.process(video_chunk)
    """

    def __init__(self, window_size: int = 50, overlap: int = 10):
        self.window_size = window_size
        self.overlap = overlap
        self._buffer = []

    def add_frame(self, frame_bytes: bytes, metadata: dict):
        """Thêm 1 frame vào buffer."""
        self._buffer.append({"frame": frame_bytes, "metadata": metadata})

    def is_ready(self) -> bool:
        """Kiểm tra buffer đã đủ N frames chưa."""
        return len(self._buffer) >= self.window_size

    def flush(self) -> dict:
        """
        Xuất Video Chunk từ buffer và giữ lại overlap frames.

        Returns:
            dict với keys: video_bytes, frames_metadata, camera_id, timestamp_range
        TODO: Implement overlay + encoding logic.
        """
        raise NotImplementedError("VideoSynthesizer.flush() chưa được implement.")
