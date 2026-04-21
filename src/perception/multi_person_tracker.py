"""多人跟踪模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PersonTrack:
    """单个跟踪目标。"""

    person_id: str
    bbox: tuple[float, float, float, float]
    embedding: list[float]
    last_seen_frame: int
    distance_m: float
    last_interaction_ts: float = 0.0


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """计算两个框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _cos_sim(a: list[float], b: list[float]) -> float:
    """计算 embedding 余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MultiPersonTracker:
    """基于 IoU + embedding 的轻量级多人跟踪器。"""

    def __init__(self, iou_threshold: float = 0.4, emb_threshold: float = 0.75) -> None:
        self.iou_threshold = iou_threshold
        self.emb_threshold = emb_threshold
        self._tracks: dict[str, PersonTrack] = {}
        self._next_id = 1

    def update(self, detections: list[dict[str, Any]], frame_index: int) -> list[PersonTrack]:
        """根据检测结果更新轨迹。"""
        updated_ids: set[str] = set()
        for det in detections:
            bbox = tuple(det.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            emb = list(det.get("embedding", []))
            distance_m = float(det.get("distance_m", 9.9))
            best_id = ""
            best_score = -1.0
            for pid, track in self._tracks.items():
                score = 0.6 * _iou(bbox, track.bbox) + 0.4 * _cos_sim(emb, track.embedding)
                if score > best_score:
                    best_score = score
                    best_id = pid
            if best_id and best_score >= (0.6 * self.iou_threshold + 0.4 * self.emb_threshold):
                track = self._tracks[best_id]
                track.bbox = bbox
                track.embedding = emb
                track.distance_m = distance_m
                track.last_seen_frame = frame_index
                updated_ids.add(best_id)
            else:
                pid = f"p{self._next_id}"
                self._next_id += 1
                self._tracks[pid] = PersonTrack(
                    person_id=pid,
                    bbox=bbox,
                    embedding=emb,
                    last_seen_frame=frame_index,
                    distance_m=distance_m,
                )
                updated_ids.add(pid)
        stale_ids = [pid for pid, track in self._tracks.items() if frame_index - track.last_seen_frame > 30]
        for pid in stale_ids:
            self._tracks.pop(pid, None)
        return [self._tracks[pid] for pid in self._tracks if pid in updated_ids]
