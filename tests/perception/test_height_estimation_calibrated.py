"""标定身高估计测试。"""

from src.perception.pose_height_estimator import PoseHeightEstimator


def test_height_estimator_should_return_zero_on_empty_frame() -> None:
    """空帧应返回 0。"""
    estimator = PoseHeightEstimator()
    result = estimator.estimate(b"")
    assert result.height_cm == 0.0
    assert result.confidence == 0.0


def test_height_estimator_should_expose_kpt_coverage() -> None:
    """估计结果应包含关键点覆盖度。"""
    estimator = PoseHeightEstimator()
    result = estimator.estimate(b"not-an-image")
    assert result.kpt_coverage >= 0.0
