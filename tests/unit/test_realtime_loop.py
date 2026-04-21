"""RealtimeLoop 单元测试。"""

from src.orchestration.realtime_loop import RealtimeLoop


def test_run_one_turn_should_include_latency_metrics() -> None:
    """单轮执行结果应包含性能指标。"""
    loop = RealtimeLoop()

    result = loop.run_one_turn(
        audio_chunks=[b"\xe4\xbd\xa0\xe5\xa5\xbd", b"hello"],
        vision_summary="",
        frame_stream=[b"frame-a"],
    )

    assert "metrics" in result
    assert "latency_ms" in result["metrics"]
    assert result["metrics"]["latency_ms"]["total"] >= 0
    assert "degraded" in result["metrics"]


def test_stream_input_events_should_limit_vision_frames() -> None:
    """视觉处理应支持限流，避免高帧率导致时延放大。"""
    loop = RealtimeLoop()

    result = loop.stream_input_events(
        frame_stream=[b"f1", b"f2", b"f3", b"f4", b"f5"],
        audio_chunks=[b"hi"],
        max_vision_frames=2,
    )

    assert result["vision_frames_processed"] == 2
