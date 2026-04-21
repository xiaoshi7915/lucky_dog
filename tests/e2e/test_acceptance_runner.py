"""真机实连前压测验收运行器测试。"""

from src.orchestration.acceptance_runner import run_duplex_acceptance
from src.orchestration.realtime_loop import RealtimeLoop
from src.perception.camera_input import CameraInput


def test_run_duplex_acceptance_should_collect_metrics_and_interruptions() -> None:
    """应统计回合数、中断次数和首响应中位数。"""
    loop = RealtimeLoop()
    report = run_duplex_acceptance(
        loop=loop,
        camera=CameraInput(camera_index=-1),
        duration_sec=1.0,
        min_interruptions=1,
        max_first_response_median_ms=3000.0,
    )
    assert report["turn_count"] > 0
    assert report["interruptions"] >= 1
    assert report["first_response_median_ms"] >= 0.0
    assert "vision_detection_rate" in report
    assert report["passed"] is True
