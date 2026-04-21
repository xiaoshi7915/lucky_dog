"""运行 5 个业务场景端到端验收。"""

from __future__ import annotations

import json
import sys

from src.orchestration.acceptance_runner import run_duplex_acceptance
from src.orchestration.realtime_loop import RealtimeLoop
from src.perception.camera_input import CameraInput


SCENARIOS = [
    "scenario1_single_user",
    "scenario2_multi_person",
    "scenario3_noise",
    "scenario4_barge_in",
    "scenario5_low_light",
]


def main() -> int:
    loop = RealtimeLoop()
    camera = CameraInput(camera_index=0)
    reports = []
    for scenario in SCENARIOS:
        report = run_duplex_acceptance(
            loop=loop,
            camera=camera,
            duration_sec=5.0,
            min_interruptions=1 if scenario == "scenario4_barge_in" else 0,
            max_first_response_median_ms=2500.0,
            scenario_name=scenario,
        )
        report["scenario"] = scenario
        reports.append(report)
    summary = {
        "scenario_reports": reports,
        "pass_rate": round(sum(1 for item in reports if item["passed"]) / len(reports), 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass_rate"] >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
