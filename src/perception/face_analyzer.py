"""视觉人脸分析模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import VisionConfig
from src.perception.face_quality import FaceQualityAssessor

try:
    import cv2
except ImportError:  # pragma: no cover - 依赖可选
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依赖可选
    np = None

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover - 依赖可选
    FaceAnalysis = None


@dataclass
class FaceAttributes:
    """人脸属性结构。"""

    gender: str
    age: int
    face_quality: float
    confidence: float


class FaceAnalyzer:
    """InsightFace + FQA 人脸分析器。"""

    def __init__(self, vision_config: VisionConfig | None = None) -> None:
        self.vision_config = vision_config
        self.face_conf_threshold = vision_config.thresholds.face_confidence if vision_config else 0.55
        self.face_quality_assessor = FaceQualityAssessor(
            model_path=vision_config.fqa_model_path if vision_config else "",
            confidence_threshold=vision_config.thresholds.fqa_confidence if vision_config else 0.3,
        )
        self._face_analysis = self._build_face_model(vision_config)

    @staticmethod
    def _build_face_model(vision_config: VisionConfig | None) -> Any:
        """初始化 InsightFace 模型。"""
        if FaceAnalysis is None:
            return None
        model_pack = vision_config.insightface_model_pack if vision_config else "buffalo_l"
        providers = vision_config.insightface_providers if vision_config else ["CPUExecutionProvider"]
        try:
            face_model = FaceAnalysis(name=model_pack, providers=providers)
            face_model.prepare(ctx_id=0, det_size=(640, 640))
            return face_model
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

    @staticmethod
    def _pick_primary_face(faces: list[Any]) -> Any | None:
        """多人时默认选最大人脸框。"""
        if not faces:
            return None
        return max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))

    def analyze(self, frame: Any) -> FaceAttributes:
        """分析一帧图像并返回属性。"""
        bgr_frame = self._decode_frame(frame)
        if bgr_frame is None:
            return FaceAttributes(gender="unknown", age=0, face_quality=0.0, confidence=0.0)
        if self._face_analysis is None:
            return FaceAttributes(gender="unknown", age=0, face_quality=0.0, confidence=0.0)
        faces = self._face_analysis.get(bgr_frame)
        primary_face = self._pick_primary_face(faces)
        if primary_face is None:
            return FaceAttributes(gender="unknown", age=0, face_quality=0.0, confidence=0.0)
        face_confidence = float(getattr(primary_face, "det_score", 0.0))
        if face_confidence < self.face_conf_threshold:
            return FaceAttributes(gender="unknown", age=0, face_quality=0.0, confidence=round(face_confidence, 2))
        gender_raw = int(getattr(primary_face, "gender", -1))
        gender = "female" if gender_raw == 0 else "male" if gender_raw == 1 else "unknown"
        age = int(getattr(primary_face, "age", 0))
        x1, y1, x2, y2 = [int(v) for v in primary_face.bbox]
        face_crop = bgr_frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        quality = self.face_quality_assessor.score(face_crop)
        return FaceAttributes(
            gender=gender,
            age=max(age, 0),
            face_quality=round(quality, 2),
            confidence=round(face_confidence, 2),
        )
