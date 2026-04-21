"""机器狗控制抽象层。"""

from abc import ABC, abstractmethod


class DogControllerBase(ABC):
    """动作控制器基础接口。"""

    @abstractmethod
    def execute_action(self, action_intent: str) -> bool:
        """执行动作意图并返回执行结果。"""
