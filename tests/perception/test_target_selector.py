from src.perception.multi_person_tracker import MultiPersonTracker
from src.perception.target_selector import ActiveSpeakerTargetSelector


def test_target_selector_should_prefer_direction_match() -> None:
    tracker = MultiPersonTracker()
    tracks = tracker.update(
        [
            {"bbox": (0.1, 0.2, 0.3, 0.8), "embedding": [1.0, 0.0], "distance_m": 1.0},
            {"bbox": (0.7, 0.2, 0.9, 0.8), "embedding": [0.0, 1.0], "distance_m": 0.8},
        ],
        frame_index=1,
    )
    selector = ActiveSpeakerTargetSelector()
    selection = selector.select(tracks, speaker_direction="left")
    assert selection.active_person_id
    assert "direction_match" in selection.reason
