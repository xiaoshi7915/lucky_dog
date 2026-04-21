"""流式语音识别模块。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from src.perception.audio_input import MicrophonePCMStream, is_speech_chunk, normalize_pcm16_mono


@dataclass
class ASRPartialResult:
    """ASR 分片识别结果。"""

    text: str
    is_final: bool
    confidence: float
    chunk_index: int


class FunASRBackendProtocol(Protocol):
    """FunASR 后端协议。"""

    def transcribe_stream(self, pcm_chunk: bytes) -> dict[str, Any]:
        """输入一个 chunk 并返回识别结果。"""


class _FunASRBackend:
    """FunASR SDK 封装。"""

    def __init__(self, model_name: str) -> None:
        self._is_available = False
        self._model: Any | None = None
        self._model_name = model_name
        try:
            from funasr import AutoModel  # type: ignore

            self._model = AutoModel(model=model_name)
            self._is_available = True
        except Exception:
            self._is_available = False

    def transcribe_stream(self, pcm_chunk: bytes) -> dict[str, Any]:
        """调用 FunASR 流式推理。"""
        if not self._is_available or self._model is None:
            return {"text": "", "is_final": False, "confidence": 0.0}
        result = self._model.generate(input=pcm_chunk, is_final=False)
        text = ""
        if isinstance(result, list) and result:
            text = str(result[0].get("text", "")).strip()
        elif isinstance(result, dict):
            text = str(result.get("text", "")).strip()
        return {"text": text, "is_final": False, "confidence": 0.9 if text else 0.0}

    @property
    def is_available(self) -> bool:
        """返回 FunASR 后端就绪状态。"""
        return self._is_available


class ASRStream:
    """ASR 流式识别实现（默认 FunASR）。"""

    def __init__(
        self,
        backend: FunASRBackendProtocol | None = None,
        model_name: str = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404",
        sample_rate: int = 16000,
        chunk_ms: int = 200,
        vad_threshold: float = 0.01,
        channels: int = 1,
    ) -> None:
        self.backend = backend or _FunASRBackend(model_name=model_name)
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.vad_threshold = vad_threshold
        self.channels = channels
        self.min_silence_ms = 300

    @property
    def is_ready(self) -> bool:
        """返回 ASR 组件是否就绪。"""
        return bool(getattr(self.backend, "is_available", True))

    def iter_partial_results(self, audio_chunks: Iterable[bytes]) -> Iterable[ASRPartialResult]:
        """将音频分片逐步转为文本片段事件。"""
        for chunk_index, chunk in enumerate(audio_chunks):
            pcm_chunk = normalize_pcm16_mono(chunk, channels=self.channels)
            if not is_speech_chunk(pcm_chunk, vad_threshold=self.vad_threshold):
                continue
            result = self.backend.transcribe_stream(pcm_chunk)
            text = str(result.get("text", "")).strip()
            if not text:
                continue
            yield ASRPartialResult(
                text=text,
                is_final=bool(result.get("is_final", False)),
                confidence=float(result.get("confidence", 0.0)),
                chunk_index=chunk_index,
            )

    def transcribe_chunks(self, audio_chunks: Iterable[bytes]) -> str:
        """聚合分片识别结果得到单轮文本。"""
        partial_texts = [result.text for result in self.iter_partial_results(audio_chunks)]
        return " ".join(partial_texts).strip()

    def run_live_stream(self, duration_sec: float = 5.0) -> dict[str, float | int]:
        """运行真实麦克风流式识别并返回延迟指标。"""
        mic = MicrophonePCMStream(sample_rate=self.sample_rate, channels=self.channels, chunk_ms=self.chunk_ms)
        if not mic.is_available:
            raise RuntimeError("缺少 sounddevice，无法运行真实 ASR 麦克风链路。")
        start = perf_counter()
        first_partial_ts = 0.0
        final_ts = 0.0
        empty_count = 0
        chunk_count = 0
        for chunk_count, chunk in enumerate(mic.iter_chunks(duration_sec=duration_sec), start=1):
            pcm_chunk = normalize_pcm16_mono(chunk, channels=self.channels)
            if not is_speech_chunk(pcm_chunk, vad_threshold=self.vad_threshold):
                empty_count += 1
                continue
            result = self.backend.transcribe_stream(pcm_chunk)
            text = str(result.get("text", "")).strip()
            if not text:
                empty_count += 1
                continue
            now = perf_counter()
            if first_partial_ts <= 0:
                first_partial_ts = now
            if bool(result.get("is_final", False)):
                final_ts = now
        end = perf_counter()
        return {
            "chunks": chunk_count,
            "first_token_latency_ms": round(((first_partial_ts or end) - start) * 1000, 2),
            "final_latency_ms": round(((final_ts or end) - start) * 1000, 2),
            "empty_ratio": round(empty_count / max(chunk_count, 1), 4),
        }
