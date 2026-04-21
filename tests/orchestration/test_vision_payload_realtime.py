"""实时视觉载荷测试。"""

from src.orchestration.realtime_loop import RealtimeLoop


def test_vision_payload_should_include_new_fields() -> None:
    """视觉载荷应包含统一字段。"""
    loop = RealtimeLoop()
    payload = loop._build_vision_payload(b"")
    assert "beauty_score_0_10" in payload
    assert "vision_confidence" in payload
    assert "uncertainty_note" in payload
    assert "tracked_person_count" in payload
    assert payload.get("speaker_direction") in {"left", "right", "center"}
