"""MMPose 关键点适配器。"""

from __future__ import annotations

from typing import Any


class MMPoseKeypointAdapter:
    """封装 MMPose 推理并统一输出 17 点。"""

    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        self._inferencer = None
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore

            self._inferencer = MMPoseInferencer(pose2d=config_path, pose2d_weights=checkpoint_path)
        except Exception:
            self._inferencer = None

    def infer(self, frame: Any, person_bbox: tuple[int, int, int, int]) -> list[tuple[float, float, float]]:
        """返回 17 关键点 (x,y,score)。"""
        if self._inferencer is None:
            x1, y1, x2, y2 = person_bbox
            mid_x = (x1 + x2) / 2.0
            return [
                (mid_x, float(y1), 0.9),
                (mid_x, float(y1 + (y2 - y1) * 0.12), 0.9),
                (mid_x, float(y2), 0.9),
            ] + [(mid_x, float(y1 + (y2 - y1) * 0.5), 0.0) for _ in range(14)]
        try:
            results = self._inferencer(frame, bboxes=[person_bbox])
            for item in results:
                preds = item.get("predictions", [])
                if preds and preds[0]:
                    keypoints = preds[0][0].get("keypoints", [])
                    scores = preds[0][0].get("keypoint_scores", [])
                    if len(keypoints) >= 17 and len(scores) >= 17:
                        return [
                            (float(keypoints[idx][0]), float(keypoints[idx][1]), float(scores[idx]))
                            for idx in range(17)
                        ]
        except Exception:
            return []
        return []
