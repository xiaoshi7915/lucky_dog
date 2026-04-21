"""一键执行全双工验收脚本。"""

from __future__ import annotations

import argparse
import json
import sys

from src.orchestration.acceptance_runner import run_duplex_acceptance
from src.orchestration.realtime_loop import RealtimeLoop
from src.perception.camera_input import CameraInput


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行全双工 5 分钟验收压测")
    parser.add_argument("--duration-sec", type=float, default=300.0, help="压测总时长（秒）")
    parser.add_argument("--min-interruptions", type=int, default=1, help="最少插话中断次数")
    parser.add_argument(
        "--max-first-response-median-ms",
        type=float,
        default=2500.0,
        help="首响应中位数上限（毫秒）",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引")
    parser.add_argument("--scenario", type=str, default="multi_person_rotation", help="验收场景名称")
    return parser.parse_args()


def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    loop = RealtimeLoop()
    camera = CameraInput(camera_index=args.camera_index)
    report = run_duplex_acceptance(
        loop=loop,
        camera=camera,
        duration_sec=args.duration_sec,
        min_interruptions=args.min_interruptions,
        max_first_response_median_ms=args.max_first_response_median_ms,
    )
    report["scenario"] = args.scenario
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("passed", False):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
