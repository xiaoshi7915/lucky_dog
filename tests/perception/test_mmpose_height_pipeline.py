from src.config import load_config
from src.perception.pose_height_estimator import PoseHeightEstimator


def test_mmpose_pipeline_should_output_non_negative_coverage() -> None:
    config = load_config("configs/app.yaml", "configs/models.yaml")
    estimator = PoseHeightEstimator(config.vision)
    estimate = estimator.estimate(b"")
    assert estimate.kpt_coverage >= 0.0
