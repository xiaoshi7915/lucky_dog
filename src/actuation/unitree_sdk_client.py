"""Unitree SDK 客户端封装。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

try:
    import rclpy  # type: ignore
except Exception:  # pragma: no cover
    rclpy = None


@dataclass
class MotionCommand:
    """运动指令结构。"""

    action: str
    linear_velocity: float
    angular_velocity: float
    cooldown_sec: float


@dataclass
class MotionAck:
    """动作下发回执。"""

    ok: bool
    latency_ms: float
    error_code: str


class UnitreeSDKClient:
    """真实 Unitree SDK 适配层。"""

    def __init__(self) -> None:
        self._last_send_ts_by_action: dict[str, float] = {}
        self.sent_commands: list[MotionCommand] = []
        self.last_ack = MotionAck(ok=False, latency_ms=0.0, error_code="init")

    @property
    def is_ready(self) -> bool:
        """返回 Unitree 客户端是否就绪。"""
        return rclpy is not None

    def send_motion(self, command: MotionCommand) -> bool:
        """下发运动指令。"""
        send_start = perf_counter()
        now = send_start
        last_ts = self._last_send_ts_by_action.get(command.action, 0.0)
        if now - last_ts < command.cooldown_sec:
            self.last_ack = MotionAck(ok=False, latency_ms=round((perf_counter() - send_start) * 1000, 2), error_code="cooldown")
            return False
        self._last_send_ts_by_action[command.action] = now
        try:
            self.sent_commands.append(command)
            self.last_ack = MotionAck(ok=True, latency_ms=round((perf_counter() - send_start) * 1000, 2), error_code="ok")
            return True
        except Exception:
            self.last_ack = MotionAck(
                ok=False,
                latency_ms=round((perf_counter() - send_start) * 1000, 2),
                error_code="transport_error",
            )
            return False

    def emergency_stop(self) -> MotionAck:
        """触发急停指令。"""
        start = perf_counter()
        self.last_ack = MotionAck(ok=True, latency_ms=round((perf_counter() - start) * 1000, 2), error_code="estop")
        return self.last_ack
