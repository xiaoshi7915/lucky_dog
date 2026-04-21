"""说话方向估计模块。"""

from __future__ import annotations

import struct
from collections.abc import Iterable


class SpeakerDirectionEstimator:
    """基于双声道能量差的方向估计。"""

    def __init__(self, direction_threshold: float = 0.12) -> None:
        self.direction_threshold = direction_threshold

    @staticmethod
    def _stereo_balance(chunk: bytes) -> float:
        """返回左右声道归一化能量差。"""
        if not chunk or len(chunk) < 4:
            return 0.0
        sample_count = len(chunk) // 2
        values = struct.unpack("<" + "h" * sample_count, chunk[: sample_count * 2])
        left = values[0::2]
        right = values[1::2]
        if not left or not right:
            return 0.0
        left_energy = sum(v * v for v in left) / len(left)
        right_energy = sum(v * v for v in right) / len(right)
        total = left_energy + right_energy
        if total <= 0:
            return 0.0
        return (left_energy - right_energy) / total

    def estimate(self, audio_chunks: Iterable[bytes], channels: int = 1) -> str:
        """输出 left/right/center。"""
        if channels < 2:
            return "center"
        balances = [self._stereo_balance(chunk) for chunk in audio_chunks if chunk]
        if not balances:
            return "center"
        avg_balance = sum(balances) / len(balances)
        if avg_balance >= self.direction_threshold:
            return "left"
        if avg_balance <= -self.direction_threshold:
            return "right"
        return "center"
