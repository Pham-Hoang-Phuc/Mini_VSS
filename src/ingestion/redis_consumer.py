"""
redis_consumer.py
=================
Module nhận dữ liệu từ Edge Box qua Redis Streams.

Pipeline (theo vss_context.md - Mục 2):
  - Đọc liên tục từ Redis Streams
  - Mỗi message gồm: 1 Frame (.jpg/binary) + 1 Metadata (.json)
  - Metadata chứa: camera_id, frame_id, ts_system, objects (bbox, conf, cls_name...)

TODO:
  - Implement RedisConsumer class kết nối Redis Streams
  - Xử lý deserialize frame + metadata
  - Forward sang processing pipeline (video_synthesizer)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DetectedObject:
    """Một đối tượng được phát hiện trong frame (từ Edge Box)."""
    obj_id: int
    cls_name: str
    bbox: List[float]
    conf: float
    objects: List[dict] = field(default_factory=list)  # sub-objects (face, helmet...)


@dataclass
class FrameMetadata:
    """Metadata đi kèm mỗi frame từ Edge Box."""
    camera_id: int
    frame_id: int
    ts_system: float
    objects: List[DetectedObject] = field(default_factory=list)


class RedisConsumer:
    """
    Đọc Frame + Metadata từ Redis Streams (Edge Box).

    Usage tương lai:
        consumer = RedisConsumer(redis_url="redis://localhost:6379", stream_name="vss:frames")
        for frame, metadata in consumer.consume():
            video_synthesizer.add_frame(frame, metadata)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", stream_name: str = "vss:frames"):
        self.redis_url = redis_url
        self.stream_name = stream_name
        # TODO: self.client = redis.Redis.from_url(redis_url)

    def consume(self):
        """
        Generator: yield (frame_bytes, FrameMetadata) từ Redis Streams.
        TODO: Implement XREADGROUP logic.
        """
        raise NotImplementedError("RedisConsumer.consume() chưa được implement.")
