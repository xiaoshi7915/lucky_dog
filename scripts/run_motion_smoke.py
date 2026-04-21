"""运行 Unitree 动作下发冒烟脚本。"""

from __future__ import annotations

import json
import sys

from src.actuation.dog_controller_unitree import DogControllerUnitree


def main() -> int:
    controller = DogControllerUnitree()
    actions = ["nod_head", "turn_left", "turn_right", "approach", "step_back", "wag_tail"]
    result = []
    for action in actions:
        ok = controller.execute_action(action)
        result.append({"action": action, "ok": ok, "reason": controller.execution_history[-1].reason})
    print(json.dumps({"results": result}, ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in result) else 1


if __name__ == "__main__":
    sys.exit(main())
