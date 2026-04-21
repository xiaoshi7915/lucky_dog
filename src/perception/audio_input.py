"""音频输入与预处理工具。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import struct
from time import sleep

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None


def normalize_pcm16_mono(chunk: bytes, channels: int = 1) -> bytes:
    """将输入音频标准化为 16bit 单声道 PCM。"""
    if channels <= 1:
        return chunk
    sample_count = len(chunk) // 2
    values = struct.unpack("<" + "h" * sample_count, chunk[: sample_count * 2])
    mono_samples: list[int] = []
    for index in range(0, len(values), channels):
        frame = values[index : index + channels]
        mono_samples.append(int(sum(frame) / len(frame)))
    return struct.pack("<" + "h" * len(mono_samples), *mono_samples)


def rms_energy(chunk: bytes) -> float:
    """计算 PCM chunk 的 RMS 能量。"""
    if not chunk:
        return 0.0
    sample_count = len(chunk) // 2
    if sample_count == 0:
        return 0.0
    values = struct.unpack("<" + "h" * sample_count, chunk[: sample_count * 2])
    square_sum = sum(value * value for value in values)
    rms_value = (square_sum / sample_count) ** 0.5
    return float(rms_value) / float(2**15)


def is_speech_chunk(chunk: bytes, vad_threshold: float) -> bool:
    """根据能量阈值判断是否语音片段。"""
    return rms_energy(chunk) >= max(vad_threshold, 0.0)


def iter_fixed_chunks(audio_bytes: bytes, sample_rate: int, chunk_ms: int) -> Iterable[bytes]:
    """将音频切分为固定时长分片。"""
    bytes_per_sample = 2
    chunk_size = max(1, int(sample_rate * chunk_ms / 1000) * bytes_per_sample)
    for start in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[start : start + chunk_size]


class MicrophonePCMStream:
    """真实麦克风 PCM 分片流。"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_ms: int = 200) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms

    @property
    def is_available(self) -> bool:
        """返回真实麦克风依赖是否可用。"""
        return sd is not None

    def iter_chunks(self, duration_sec: float) -> Iterator[bytes]:
        """按时长产出真实麦克风 chunk（16bit PCM）。"""
        if sd is None:
            raise RuntimeError("缺少 sounddevice，无法读取真实麦克风。")
        chunk_frames = max(1, int(self.sample_rate * self.chunk_ms / 1000))
        total_chunks = max(1, int(duration_sec * 1000 / self.chunk_ms))
        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=chunk_frames,
        )
        stream.start()
        try:
            for _ in range(total_chunks):
                data, overflowed = stream.read(chunk_frames)
                if overflowed:
                    sleep(0.001)
                yield bytes(data)
        finally:
            stream.stop()
            stream.close()
