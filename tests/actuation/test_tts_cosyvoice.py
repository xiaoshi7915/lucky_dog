"""CosyVoice TTS 行为测试。"""

from src.actuation.tts_engine import TTSEngine


def test_stream_synthesize_should_emit_audio_chunks() -> None:
    """流式合成应持续输出音频块。"""
    engine = TTSEngine(provider="cosyvoice")
    chunks = list(engine.stream_synthesize(["你", "好"]))
    assert len(chunks) == 2
    assert all(isinstance(item, bytes) and item for item in chunks)


def test_interrupt_should_stop_playback_queue() -> None:
    """打断时应清空播放队列并停止播放状态。"""
    engine = TTSEngine(provider="cosyvoice")
    engine.enqueue_for_playback(b"a")
    engine.enqueue_for_playback(b"b")
    assert engine.pending_audio_count == 2
    engine.interrupt_playback()
    assert engine.pending_audio_count == 0
    assert engine.is_playing is False
