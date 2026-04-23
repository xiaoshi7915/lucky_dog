"""CosyVoice TTS 行为测试。"""

from src.actuation.tts_engine import TTSEngine


def test_stream_synthesize_without_delimiter_should_emit_one_chunk() -> None:
    """无标点句子：所有 token 合并后一次合成，输出 1 个 chunk（避免逐字符多次合成）。"""
    engine = TTSEngine(provider="cosyvoice")
    chunks = list(engine.stream_synthesize(["你", "好"]))
    # 修复后：无句子边界标点时合并为单次合成，输出 1 个 chunk
    assert len(chunks) == 1
    assert all(isinstance(item, bytes) and item for item in chunks)


def test_stream_synthesize_with_delimiter_should_split_chunks() -> None:
    """含标点句子：遇到句子边界标点立即合成，产出多个 chunk。"""
    engine = TTSEngine(provider="cosyvoice")
    # "你好。" 遇到句号合成第一块，"再见" 在末尾合成第二块
    chunks = list(engine.stream_synthesize(["你好", "。", "再见"]))
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
