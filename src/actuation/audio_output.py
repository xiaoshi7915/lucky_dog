"""音频输出队列封装。"""

from collections import deque
import wave
from pathlib import Path

try:
    import simpleaudio as sa  # type: ignore
except Exception:  # pragma: no cover
    sa = None


class AudioOutputQueue:
    """线程无关的轻量播放队列。"""

    def __init__(self) -> None:
        self._queue: deque[bytes] = deque()
        self._interrupted = False

    def push(self, audio_chunk: bytes) -> None:
        """写入待播放音频块。"""
        self._queue.append(audio_chunk)

    def pop(self) -> bytes | None:
        """弹出下一段音频。"""
        if not self._queue:
            return None
        return self._queue.popleft()

    def clear(self) -> None:
        """清空队列。"""
        self._queue.clear()
        self._interrupted = True

    def reset_interrupt(self) -> None:
        """重置中断标志。"""
        self._interrupted = False

    @property
    def size(self) -> int:
        """返回待播放块数量。"""
        return len(self._queue)

    @property
    def interrupted(self) -> bool:
        """返回是否发生过打断。"""
        return self._interrupted


class AudioPlayer:
    """本地音频播放器。"""

    def __init__(self, sample_rate: int = 22050, channels: int = 1, sample_width: int = 2) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._play_obj = None

    @property
    def is_available(self) -> bool:
        """返回播放器依赖是否可用。"""
        return sa is not None

    def play_pcm16(self, pcm_bytes: bytes) -> bool:
        """播放 PCM16 单声道音频。"""
        if not pcm_bytes or sa is None:
            return False
        self._play_obj = sa.play_buffer(pcm_bytes, self.channels, self.sample_width, self.sample_rate)
        return True

    def stop(self) -> None:
        """停止播放。"""
        if self._play_obj is not None:
            self._play_obj.stop()

    def save_wav(self, audio_bytes: bytes, path: str | Path) -> Path:
        """将 PCM16 保存为 wav 文件。"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
        return output
