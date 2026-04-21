"""YOLO + InsightFace 联合多人检测。"""

from __future__ import annotations

from typing import Any

from src.config import VisionConfig
from src.perception.camera_calibration import CalibrationParams
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

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover
    FaceAnalysis = None


class PersonFaceDetector:
    """输出用于跟踪的统一检测结构。"""

    def __init__(self, vision_config: VisionConfig) -> None:
        self.person_threshold = vision_config.thresholds.person_confidence
        self.face_threshold = vision_config.thresholds.face_confidence
        self._calibration = CalibrationParams(
            camera_height_cm=vision_config.calibration.camera_height_cm,
            camera_tilt_deg=vision_config.calibration.camera_tilt_deg,
            focal_length_px=vision_config.calibration.focal_length_px,
        )
        self._yolo = self._build_yolo(vision_config.yolo_model_path)
        self._face = self._build_face(vision_config.insightface_model_pack, vision_config.insightface_providers)
        self._mmpose = MMPoseKeypointAdapter(
            config_path=vision_config.mmpose_config_path,
            checkpoint_path=vision_config.mmpose_checkpoint_path,
        )

    @staticmethod
    def _build_yolo(model_path: str) -> Any:
        if YOLO is None:
            return None
        try:
            return YOLO(model_path)
        except Exception:
            return None

    @staticmethod
    def _build_face(model_pack: str, providers: list[str]) -> Any:
        if FaceAnalysis is None:
            return None
        try:
            app = FaceAnalysis(name=model_pack, providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception:
            return None

    @staticmethod
    def decode_frame(frame: Any) -> Any | None:
        """将输入解码为 BGR。"""
        if np is not None and isinstance(frame, np.ndarray):
            return frame
        if not frame or np is None or cv2 is None:
            return None
        arr = np.frombuffer(frame, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _norm_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return (
            max(0.0, min(1.0, x1 / max(width, 1))),
            max(0.0, min(1.0, y1 / max(height, 1))),
            max(0.0, min(1.0, x2 / max(width, 1))),
            max(0.0, min(1.0, y2 / max(height, 1))),
        )

    @staticmethod
    def _match_face_embedding(
        person_bbox: tuple[int, int, int, int],
        faces: list[Any],
        face_threshold: float,
    ) -> list[float]:
        """将人脸 embedding 关联到对应人体框。"""
        px1, py1, px2, py2 = person_bbox
        for face in faces:
            det_score = float(getattr(face, "det_score", 0.0))
            if det_score < face_threshold:
                continue
            fx1, fy1, fx2, fy2 = [int(v) for v in getattr(face, "bbox", [0, 0, 0, 0])]
            center_x = (fx1 + fx2) / 2
            center_y = (fy1 + fy2) / 2
            if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                emb = getattr(face, "embedding", [])
                if emb is not None:
                    return [float(v) for v in list(emb)[:128]]
        return []

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """返回标准化 detections：bbox/embedding/distance_m/confidence。"""
        bgr = self.decode_frame(frame)
        if bgr is None:
            return []
        h, w = bgr.shape[:2]
        faces = self._face.get(bgr) if self._face is not None else []
        if self._yolo is None:
            return []
        try:
            result = self._yolo.predict(source=bgr, verbose=False)[0]
        except Exception:
            return []
        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if cls_id != 0 or conf < self.person_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            person_bbox = (x1, y1, x2, y2)
            keypoints = self._mmpose.infer(bgr, person_bbox)
            valid = [point for point in keypoints if len(point) == 3 and point[2] >= 0.2]
            head_y = min((point[1] for point in valid), default=float(y1))
            foot_y = max((point[1] for point in valid), default=float(y2))
            pixel_h = max(1.0, foot_y - head_y)
            # 小孔相机模型：distance ~= focal * real_height / pixel_height
            assumed_person_height_cm = 165.0
            tilt_adjust = max(0.3, abs(self._calibration.camera_tilt_deg) / 45.0 + 0.7)
            distance_cm = (self._calibration.focal_length_px * assumed_person_height_cm / pixel_h) * tilt_adjust
            distance_m = round(distance_cm / 100.0, 2)
            detections.append(
                {
                    "bbox": self._norm_bbox(person_bbox, w, h),
                    "embedding": self._match_face_embedding(person_bbox, faces, self.face_threshold),
                    "distance_m": distance_m,
                    "confidence": conf,
                    "kpt_coverage": round(len(valid) / 17.0 if keypoints else 0.0, 2),
                }
            )
        return detections
