from src.actuation.tts_engine import TTSEngine


def test_cosyvoice_barge_in_should_clear_queue() -> None:
    engine = TTSEngine(provider="cosyvoice")
    list(engine.stream_synthesize(["你好", "世界"]))
    assert engine.pending_audio_count >= 1
    engine.interrupt_playback()
    assert engine.pending_audio_count == 0
