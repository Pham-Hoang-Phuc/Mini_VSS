"""
vlm_reasoner.py
===============
Module gọi VLM (Qwen2-VL) để mô tả nội dung Video Chunk.

Pipeline (theo vss_context.md - Mục 3, Bước 2):
  - Input: Video Chunk + Context-aware Prompt
  - Nhiệm vụ: Mô tả chi tiết hành động, thuộc tính đối tượng
    (màu áo, loại thiết bị, trạng thái làm việc)
  - Output: Structured Caption (text hoặc JSON sự kiện)

TODO:
  - Kết nối Qwen2-VL API / local inference
  - Thiết kế prompt template cho context-aware description
  - Parse output thành structured format
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StructuredCaption:
    """Output từ VLM sau khi xử lý Video Chunk."""
    caption: str                    # Mô tả chi tiết bằng text
    video_chunk_url: str = ""       # URL video trên MinIO
    timestamp: str = ""             # Timestamp range (e.g. "00:00-00:30")
    camera_id: str = ""             # Camera ID
    video_id: str = ""              # Video ID


class VLMReasoner:
    """
    Gọi Qwen2-VL để sinh Structured Caption từ Video Chunk.

    Usage tương lai:
        reasoner = VLMReasoner(model_name="Qwen2-VL-7B")
        caption = reasoner.process(video_chunk_bytes, context_prompt)
    """

    def __init__(self, model_name: str = "Qwen2-VL-7B", api_url: Optional[str] = None):
        self.model_name = model_name
        self.api_url = api_url
        # TODO: Initialize model / API client

    def process(self, video_chunk: bytes, context_prompt: str = "") -> StructuredCaption:
        """
        Xử lý 1 Video Chunk → Structured Caption.

        Args:
            video_chunk: bytes của video chunk
            context_prompt: prompt bổ sung ngữ cảnh (camera location, previous events...)

        Returns:
            StructuredCaption chứa mô tả chi tiết
        TODO: Implement VLM inference.
        """
        raise NotImplementedError("VLMReasoner.process() chưa được implement.")
