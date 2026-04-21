"""全双工 RealtimeLoop 测试。"""

from src.orchestration.realtime_loop import RealtimeLoop


def test_run_duplex_turn_should_expose_duplex_metrics_and_state() -> None:
    """全双工执行结果应包含并发状态和首包延迟指标。"""
    loop = RealtimeLoop()
    result = loop.run_duplex_turn(
        audio_chunks=[b"\xe4\xbd\xa0\xe5\xa5\xbd", b"\xe8\xbf\x99\xe6\x98\xaf\xe6\xb5\x8b\xe8\xaf\x95"],
        frame_stream=[b"f1"],
    )
    assert result["state"] in {"LISTENING", "THINKING", "SPEAKING", "INTERRUPTED"}
    assert "first_token_latency_ms" in result["metrics"]
    assert "first_audio_latency_ms" in result["metrics"]


def test_run_duplex_turn_should_interrupt_tts_when_barge_in_detected() -> None:
    """用户插话时应触发中断状态。"""
    loop = RealtimeLoop()
    result = loop.run_duplex_turn(
        audio_chunks=[b"first", b"barge_in"],
        frame_stream=[],
        simulate_barge_in=True,
    )
    assert result["state"] == "INTERRUPTED"
