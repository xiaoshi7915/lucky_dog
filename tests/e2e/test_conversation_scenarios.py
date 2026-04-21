"""端到端场景测试。"""

from src.orchestration.realtime_loop import RealtimeLoop


def test_single_user_scenario_should_complete_turn() -> None:
    """单人场景应能完成闭环。"""
    loop = RealtimeLoop()

    result = loop.run_one_turn(
        audio_chunks=[b"\xe4\xbd\xa0\xe5\xa5\xbd \xe9\x9d\xa0\xe8\xbf\x91"],
        frame_stream=[b"single-person-frame"],
    )

    assert result["tts_audio_size"] > 0
    assert result["action_ok"] is True


def test_multi_person_like_scenario_should_not_crash() -> None:
    """多人场景（高帧输入）应保持稳定。"""
    loop = RealtimeLoop()

    result = loop.run_one_turn(
        audio_chunks=[b"hello everyone"],
        frame_stream=[b"p1", b"p2", b"p3", b"p4", b"p5", b"p6"],
    )

    assert result["vision_frames_processed"] <= 3
    assert result["metrics"]["latency_ms"]["total"] >= 0


def test_noisy_audio_scenario_should_degrade_gracefully() -> None:
    """噪声场景应触发可观测降级。"""
    loop = RealtimeLoop()

    result = loop.run_one_turn(
        audio_chunks=[b"   ", b""],
        frame_stream=[],
    )

    assert result["metrics"]["degraded"] is True
    assert result["dialogue"]["reply_text"]
