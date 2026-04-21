"""TTS 真实输出路径测试。"""

from src.actuation.tts_engine import TTSEngine


def test_tts_engine_should_output_non_empty_audio() -> None:
    engine = TTSEngine(provider="cosyvoice")
    audio = engine.synthesize("你好")
    assert isinstance(audio, bytes)
    assert len(audio) > 0
