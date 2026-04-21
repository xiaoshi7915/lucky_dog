import struct

from src.perception.speaker_direction_estimator import SpeakerDirectionEstimator


def test_direction_estimator_should_return_center_for_mono() -> None:
    estimator = SpeakerDirectionEstimator()
    assert estimator.estimate([b"\x00\x00\x01\x00"], channels=1) == "center"


def test_direction_estimator_should_detect_left_bias() -> None:
    estimator = SpeakerDirectionEstimator(direction_threshold=0.05)
    # 交错双声道：left=3000,right=200
    values = [3000, 200] * 64
    chunk = struct.pack("<" + "h" * len(values), *values)
    assert estimator.estimate([chunk], channels=2) == "left"
