"""摄像头输入模块。"""

from __future__ import annotations

from collections.abc import Iterator

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class CameraInput:
    """负责采集真实摄像头帧。"""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height

    def read_frames(self, max_frames: int = 1) -> Iterator[bytes]:
        """读取并编码摄像头图像帧。"""
        if cv2 is None:
            return iter(())
        capture = cv2.VideoCapture(self.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not capture.isOpened():
            return iter(())
        frames: list[bytes] = []
        try:
            for _ in range(max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                frames.append(encoded.tobytes())
        finally:
            capture.release()
        return iter(frames)
