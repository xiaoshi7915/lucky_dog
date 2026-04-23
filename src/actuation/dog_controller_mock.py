"""机器狗 mock 控制器。"""

from dataclasses import dataclass

from src.actuation.dog_controller_base import DogControllerBase


@dataclass
class ActionExecutionRecord:
    """动作执行记录。"""

    action_intent: str
    executed: bool


class DogControllerMock(DogControllerBase):
    """用于 MVP 联调的动作执行模拟器。"""

    def __init__(self) -> None:
        # step_back 与 ActionPolicy 的动作集保持一致，避免 dev 模式下永远返回 action_ok=False。
        self.supported_actions = {"wag_tail", "approach", "turn_left", "turn_right", "nod_head", "step_back"}
        self.execution_history: list[ActionExecutionRecord] = []

    def execute_action(self, action_intent: str) -> bool:
        """模拟执行动作并返回成功状态。"""
        ok = action_intent in self.supported_actions
        self.execution_history.append(ActionExecutionRecord(action_intent=action_intent, executed=ok))
        return ok
