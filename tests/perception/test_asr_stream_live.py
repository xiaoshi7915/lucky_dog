"""ASR 真实链路逻辑测试。"""

from src.perception.asr_stream import ASRStream


def test_asr_stream_exposes_ready_flag() -> None:
    stream = ASRStream()
    assert isinstance(stream.is_ready, bool)
