"""Unitree 控制器接口契约测试。"""

from src.actuation.dog_controller_unitree import DogControllerUnitree


def test_execute_action_should_support_three_safe_actions() -> None:
    """应至少支持点头/转向/尾巴动作。"""
    controller = DogControllerUnitree()
    assert controller.execute_action("nod_head") is True
    assert controller.execute_action("turn_left") is True
    assert controller.execute_action("wag_tail") is True


def test_execute_action_should_rate_limit_repeated_command() -> None:
    """短时间重复动作应被限流。"""
    controller = DogControllerUnitree(min_action_interval_sec=10.0)
    assert controller.execute_action("nod_head") is True
    assert controller.execute_action("nod_head") is False
