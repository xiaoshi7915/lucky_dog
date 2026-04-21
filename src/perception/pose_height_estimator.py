"""人体关键点与身高估计模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import VisionConfig
from src.perception.camera_calibration import CalibratedHeightEstimator, CalibrationParams
from src.perception.keypoint_adapter import MMPoseKeypointAdapter

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


@dataclass
class HeightEstimate:
    """身高估计结构。"""

    height_cm: float
    confidence: float
    kpt_coverage: float = 0.0


class PoseHeightEstimator:
    """YOLOv8 + MMPose + 标定参数身高估计器。"""

    def __init__(self, vision_config: VisionConfig | None = None) -> None:
        self.vision_config = vision_config
        self.person_threshold = vision_config.thresholds.person_confidence if vision_config else 0.45
        self.kpt_threshold = vision_config.thresholds.keypoint_confidence if vision_config else 0.35
        self._yolo_model = self._build_yolo(vision_config)
        calibration = vision_config.calibration if vision_config else None
        self._calibrated_estimator = CalibratedHeightEstimator(
            CalibrationParams(
                camera_height_cm=calibration.camera_height_cm if calibration else 95.0,
                camera_tilt_deg=calibration.camera_tilt_deg if calibration else -8.0,
                focal_length_px=calibration.focal_length_px if calibration else 930.0,
            )
        )
        config_path = vision_config.mmpose_config_path if vision_config else ""
        checkpoint_path = vision_config.mmpose_checkpoint_path if vision_config else ""
        self._mmpose = MMPoseKeypointAdapter(config_path=config_path, checkpoint_path=checkpoint_path)
        self._height_history_cm: list[float] = []

    @staticmethod
    def _build_yolo(vision_config: VisionConfig | None) -> Any:
        """加载 YOLO 模型。"""
        if YOLO is None:
            return None
        model_path = vision_config.yolo_model_path if vision_config else "models/yolov8n.pt"
        try:
            return YOLO(model_path)
        except Exception:
            return None

    @staticmethod
    def _decode_frame(frame: Any) -> Any | None:
        """将输入转换为 BGR 图像。"""
        if np is not None and isinstance(frame, np.ndarray):
            return frame
        if not frame or cv2 is None or np is None:
            return None
        np_buffer = np.frombuffer(frame, dtype=np.uint8)
        if np_buffer.size == 0:
            return None
        return cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    def _detect_person_bbox(self, bgr_frame: np.ndarray) -> tuple[int, int, int, int, float] | None:
        """检测置信度最高的人体框。"""
        if self._yolo_model is None:
            h, w = bgr_frame.shape[:2]
            return (int(w * 0.2), int(h * 0.1), int(w * 0.8), int(h * 0.95), 0.5)
        try:
            result = self._yolo_model.predict(source=bgr_frame, verbose=False)[0]
            best = None
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id != 0 or conf < self.person_threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                area = (x2 - x1) * (y2 - y1)
                if best is None or area > best[0]:
                    best = (area, x1, y1, x2, y2, conf)
            if best is None:
                return None
            _, x1, y1, x2, y2, conf = best
            return (x1, y1, x2, y2, conf)
        except Exception:
            return None

    def estimate(self, frame: Any) -> HeightEstimate:
        """根据图像帧估计身高。"""
        bgr_frame = self._decode_frame(frame)
        if bgr_frame is None:
            return HeightEstimate(height_cm=0.0, confidence=0.0, kpt_coverage=0.0)
        bbox = self._detect_person_bbox(bgr_frame)
        if bbox is None:
            return HeightEstimate(height_cm=0.0, confidence=0.0, kpt_coverage=0.0)
        x1, y1, x2, y2, person_conf = bbox
        keypoints = self._mmpose.infer(bgr_frame, (x1, y1, x2, y2))
        valid_points = [k for k in keypoints if len(k) == 3 and k[2] >= self.kpt_threshold]
        kpt_coverage = (len(valid_points) / 17.0) if keypoints else 0.0
        head_y = min((k[1] for k in valid_points), default=float(y1))
        foot_y = max((k[1] for k in valid_points), default=float(y2))
        pixel_height = max(0.0, foot_y - head_y)
        estimated_height = self._calibrated_estimator.estimate_height_cm(pixel_height)
        self._height_history_cm.append(estimated_height)
        if len(self._height_history_cm) > 5:
            self._height_history_cm.pop(0)
        smoothed_height = sum(self._height_history_cm) / max(1, len(self._height_history_cm))
        confidence = min(0.99, person_conf * (0.7 + 0.3 * kpt_coverage))
        return HeightEstimate(
            height_cm=round(smoothed_height, 1),
            confidence=round(confidence if confidence >= self.kpt_threshold else 0.0, 2),
            kpt_coverage=round(kpt_coverage, 2),
        )
