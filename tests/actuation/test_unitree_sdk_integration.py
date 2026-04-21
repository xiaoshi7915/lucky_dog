"""Unitree SDK 集成行为测试。"""

from src.actuation.dog_controller_unitree import DogControllerUnitree


def test_unitree_controller_should_cover_required_actions() -> None:
    controller = DogControllerUnitree()
    for action in ["nod_head", "turn_left", "turn_right", "approach", "step_back", "wag_tail"]:
        assert controller.execute_action(action) is True
