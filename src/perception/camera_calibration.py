"""相机标定参数与身高换算工具。"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians


@dataclass
class CalibrationParams:
    """用于像素高度换算的标定参数。"""

    camera_height_cm: float
    camera_tilt_deg: float
    focal_length_px: float


class CalibratedHeightEstimator:
    """基于相机标定的身高估计器。"""

    def __init__(self, params: CalibrationParams) -> None:
        self.params = params

    def estimate_height_cm(self, pixel_height: float) -> float:
        """将像素高度换算为厘米。"""
        if pixel_height <= 0:
            return 0.0
        tilt_factor = max(0.2, cos(radians(abs(self.params.camera_tilt_deg))))
        scale = (self.params.camera_height_cm * self.params.focal_length_px) / max(pixel_height, 1.0)
        return max(0.0, scale * tilt_factor)
