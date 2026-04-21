"""人脸质量评估模块。"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - 依赖可选
    np = None

try:
    import cv2
except ImportError:  # pragma: no cover - 依赖可选
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - 依赖可选
    ort = None


class FaceQualityAssessor:
    """将 FQA 结果映射为 0~10 分。"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.3) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._session = self._build_session(model_path)

    @staticmethod
    def _build_session(model_path: str):
        """创建 ONNX 推理会话。"""
        if not model_path or ort is None:
            return None
        try:
            return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception:
            return None

    @staticmethod
    def _fallback_quality(face_crop: Any) -> float:
        """无模型时使用清晰度/亮度近似质量。"""
        if cv2 is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray)) if np is not None else 0.0
        sharp_score = min(lap_var / 500.0, 1.0)
        light_score = max(0.0, min((brightness - 40.0) / 120.0, 1.0))
        return (sharp_score * 0.7 + light_score * 0.3) * 10.0

    def score(self, face_crop: Any) -> float:
        """评估人脸质量并映射到 0~10。"""
        if face_crop.size == 0:
            return 0.0
        if self._session is None:
            return round(self._fallback_quality(face_crop), 2)
        try:
            image = cv2.resize(face_crop, (112, 112)).astype(np.float32) / 255.0 if cv2 is not None and np is not None else None
            if image is None:
                return round(self._fallback_quality(face_crop), 2)
            tensor = np.transpose(image, (2, 0, 1))[None, ...]
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: tensor})[0]
            raw_score = float(np.squeeze(output))
            normalized = max(0.0, min(raw_score, 1.0))
            if normalized < self.confidence_threshold:
                return 0.0
            return round(normalized * 10.0, 2)
        except Exception:
            return round(self._fallback_quality(face_crop), 2)
