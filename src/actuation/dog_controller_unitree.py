"""Unitree 动作控制器实现。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from src.actuation.dog_controller_base import DogControllerBase
from src.actuation.unitree_sdk_client import MotionCommand, UnitreeSDKClient


@dataclass
class UnitreeActionRecord:
    """Unitree 动作执行记录。"""

    action_intent: str
    executed: bool
    reason: str


class DogControllerUnitree(DogControllerBase):
    """Unitree 控制器，包含限流与互斥保护。"""

    def __init__(self, min_action_interval_sec: float = 0.3, max_linear_velocity: float = 0.4, max_angular_velocity: float = 0.8) -> None:
        self.min_action_interval_sec = min_action_interval_sec
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.supported_actions = {"nod_head", "turn_left", "turn_right", "wag_tail", "step_back", "approach"}
        self.execution_history: list[UnitreeActionRecord] = []
        self._last_action = ""
        self._last_action_time = 0.0
        self._client = UnitreeSDKClient()

    def _map_action(self, action_intent: str) -> MotionCommand:
        """将语义动作映射为 Unitree 指令。"""
        if action_intent == "turn_left":
            return MotionCommand(action=action_intent, linear_velocity=0.0, angular_velocity=0.5, cooldown_sec=0.3)
        if action_intent == "turn_right":
            return MotionCommand(action=action_intent, linear_velocity=0.0, angular_velocity=-0.5, cooldown_sec=0.3)
        if action_intent == "approach":
            return MotionCommand(action=action_intent, linear_velocity=0.3, angular_velocity=0.0, cooldown_sec=0.4)
        if action_intent == "step_back":
            return MotionCommand(action=action_intent, linear_velocity=-0.2, angular_velocity=0.0, cooldown_sec=0.4)
        return MotionCommand(action=action_intent, linear_velocity=0.0, angular_velocity=0.0, cooldown_sec=0.2)

    def _apply_safety_limits(self, command: MotionCommand) -> MotionCommand:
        """对线速度/角速度进行安全限幅。"""
        linear = max(-self.max_linear_velocity, min(self.max_linear_velocity, command.linear_velocity))
        angular = max(-self.max_angular_velocity, min(self.max_angular_velocity, command.angular_velocity))
        return MotionCommand(
            action=command.action,
            linear_velocity=linear,
            angular_velocity=angular,
            cooldown_sec=max(command.cooldown_sec, self.min_action_interval_sec),
        )

    def execute_action(self, action_intent: str) -> bool:
        """执行动作并返回是否成功。"""
        now = perf_counter()
        if action_intent not in self.supported_actions:
            self.execution_history.append(
                UnitreeActionRecord(action_intent=action_intent, executed=False, reason="unsupported_action")
            )
            return False
        if self._last_action == action_intent and (now - self._last_action_time) < self.min_action_interval_sec:
            self.execution_history.append(
                UnitreeActionRecord(action_intent=action_intent, executed=False, reason="rate_limited")
            )
            return False
        self._last_action = action_intent
        self._last_action_time = now
        cmd = self._apply_safety_limits(self._map_action(action_intent))
        ok = self._client.send_motion(cmd)
        if not ok and self._client.last_ack.error_code not in {"cooldown"}:
            self._client.emergency_stop()
        self.execution_history.append(
            UnitreeActionRecord(
                action_intent=action_intent,
                executed=ok,
                reason=f"{self._client.last_ack.error_code}:{self._client.last_ack.latency_ms}",
            )
        )
        return ok
