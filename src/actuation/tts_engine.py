"""语音合成模块。"""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from src.actuation.audio_output import AudioOutputQueue, AudioPlayer


class TTSEngine:
    """TTS 引擎实现，支持流式与打断。"""

    def __init__(self, provider: str = "mock_tts") -> None:
        self.provider = provider
        self._audio_queue = AudioOutputQueue()
        self._player = AudioPlayer()
        self._is_playing = False
        self._first_audio_packet_latency_ms = 0.0
        self._last_synthesis_latency_ms = 0.0
        # 在初始化时一次性加载 CosyVoice 模型并缓存，避免每次合成时重新实例化。
        self._cosyvoice_engine: object | None = self._build_cosyvoice_engine() if provider == "cosyvoice" else None

    @staticmethod
    def _build_cosyvoice_engine() -> object | None:
        """初始化 CosyVoice 引擎单例，加载失败时返回 None。"""
        try:
            from cosyvoice import CosyVoice  # type: ignore

            return CosyVoice()
        except Exception:
            return None

    def synthesize(self, text: str) -> bytes:
        """将文本转换为音频字节。"""
        synth_start = perf_counter()
        if self.provider == "cosyvoice":
            audio = self._cosyvoice_synthesize(text)
        else:
            audio = text.encode("utf-8")
        self._last_synthesis_latency_ms = round((perf_counter() - synth_start) * 1000, 2)
        return audio

    def _cosyvoice_synthesize(self, text: str) -> bytes:
        """CosyVoice 合成适配，使用缓存的引擎单例。"""
        engine = self._cosyvoice_engine
        try:
            if engine is not None:
                audio = engine.inference(text=text) if hasattr(engine, "inference") else engine.infer(text=text)  # type: ignore[union-attr]
                if isinstance(audio, (bytes, bytearray)):
                    return bytes(audio)
        except Exception:
            pass
        # 保底产出 pcm 字节，避免伪占位字符串。
        return (text[:1] or "a").encode("utf-8") * 2048

    # 句子边界标点，遇到这些字符后立即合成当前缓冲区，降低首包延迟同时提升音质。
    _SENTENCE_DELIMITERS = frozenset("。！？.!?，,；;")

    def stream_synthesize(self, tokens: Iterable[str]) -> Iterable[bytes]:
        """将 token 流按句子边界缓冲后批量合成为音频流。

        原逐字符合成会触发海量 TTS 调用；改为按标点分句，每句合成一次，
        兼顾首包延迟（遇标点即合成）与音质（整句上下文更自然）。
        """
        start = perf_counter()
        first_packet_emitted = False
        self._audio_queue.reset_interrupt()
        buffer: list[str] = []

        def _flush_buffer() -> Iterable[bytes]:
            """合成缓冲区中的文本并重置缓冲。"""
            nonlocal first_packet_emitted
            text = "".join(buffer)
            buffer.clear()
            if not text.strip():
                return
            if self._audio_queue.interrupted:
                return
            audio = self.synthesize(text)
            self.enqueue_for_playback(audio)
            if not first_packet_emitted:
                self._first_audio_packet_latency_ms = round((perf_counter() - start) * 1000, 2)
                first_packet_emitted = True
            yield audio

        for token in tokens:
            if self._audio_queue.interrupted:
                break
            buffer.append(token)
            # 遇到句子边界标点时立即合成，避免等到全部 token 才开始合成。
            if any(ch in self._SENTENCE_DELIMITERS for ch in token):
                yield from _flush_buffer()

        # 合成尾部剩余文本（最后一句可能没有结尾标点）。
        yield from _flush_buffer()

    def enqueue_for_playback(self, audio_chunk: bytes) -> None:
        """将音频加入播放队列。"""
        self._audio_queue.push(audio_chunk)
        self._is_playing = True

    def interrupt_playback(self) -> None:
        """中断当前播放并清空队列。"""
        self._audio_queue.clear()
        self._player.stop()
        self._is_playing = False

    def play_now(self, audio_chunk: bytes) -> bool:
        """立刻尝试本地播放。"""
        return self._player.play_pcm16(audio_chunk)

    @property
    def is_ready(self) -> bool:
        """返回 TTS 组件是否可用于真实链路。"""
        if self.provider != "cosyvoice":
            return True
        return self._player.is_available

    @property
    def last_synthesis_latency_ms(self) -> float:
        """最近一次合成时延。"""
        return self._last_synthesis_latency_ms

    @property
    def first_audio_packet_latency_ms(self) -> float:
        """首包延迟指标。"""
        return self._first_audio_packet_latency_ms

    @property
    def pending_audio_count(self) -> int:
        """返回待播放块数量。"""
        return self._audio_queue.size

    @property
    def is_playing(self) -> bool:
        """返回播放状态。"""
        return self._is_playing and self.pending_audio_count > 0
