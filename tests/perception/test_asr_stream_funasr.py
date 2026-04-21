"""FunASR 流式 ASR 基础测试。"""

from __future__ import annotations

import struct

from src.perception.asr_stream import ASRPartialResult, ASRStream


class _FakeFunASRBackend:
    """用于测试的 FunASR 后端假实现。"""

    def transcribe_stream(self, pcm_chunk: bytes) -> dict[str, object]:
        """根据输入 chunk 模拟 partial/final。"""
        if len(pcm_chunk) < 4:
            return {"text": "", "is_final": False, "confidence": 0.0}
        if len(pcm_chunk) > 20:
            return {"text": "你好", "is_final": True, "confidence": 0.92}
        return {"text": "你", "is_final": False, "confidence": 0.85}


def _pcm_chunk(samples: int, amp: int) -> bytes:
    """构造 16bit PCM chunk。"""
    return b"".join(struct.pack("<h", amp) for _ in range(samples))


def test_iter_partial_results_should_emit_streaming_events() -> None:
    """应输出 partial 与 final 结果，并过滤静音。"""
    stream = ASRStream(backend=_FakeFunASRBackend(), sample_rate=16000, chunk_ms=200, vad_threshold=0.005)
    chunks = [
        _pcm_chunk(3200, 0),  # 静音，应该被过滤
        _pcm_chunk(8, 2000),  # partial
        _pcm_chunk(16, 2500),  # final
    ]

    results = list(stream.iter_partial_results(chunks))

    assert len(results) == 2
    assert isinstance(results[0], ASRPartialResult)
    assert results[0].text == "你"
    assert results[0].is_final is False
    assert results[1].text == "你好"
    assert results[1].is_final is True


def test_transcribe_chunks_should_aggregate_stream_text() -> None:
    """应聚合流式文本为最终字符串。"""
    stream = ASRStream(backend=_FakeFunASRBackend(), sample_rate=16000, chunk_ms=200, vad_threshold=0.005)
    chunks = [_pcm_chunk(8, 2000), _pcm_chunk(16, 2500)]

    text = stream.transcribe_chunks(chunks)

    assert text == "你 你好"
