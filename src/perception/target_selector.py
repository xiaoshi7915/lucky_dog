"""主目标选择策略。"""

from __future__ import annotations

from dataclasses import dataclass

from src.perception.multi_person_tracker import PersonTrack


@dataclass
class TargetSelection:
    """主目标选择结果。"""

    active_person_id: str
    reason: str


class ActiveSpeakerTargetSelector:
    """按说话方向、距离、最近交互选择主目标。"""

    def __init__(self) -> None:
        self._last_active_person_id = ""

    def select(self, tracks: list[PersonTrack], speaker_direction: str = "") -> TargetSelection:
        """返回当前主目标与切换原因。"""
        if not tracks:
            return TargetSelection(active_person_id="", reason="no_tracks")
        matched: list[PersonTrack] = []
        if speaker_direction:
            for track in tracks:
                x1, _, x2, _ = track.bbox
                center_x = (x1 + x2) / 2.0
                if speaker_direction == "left" and center_x < 0.5:
                    matched.append(track)
                if speaker_direction == "right" and center_x >= 0.5:
                    matched.append(track)
                if speaker_direction == "center" and 0.35 <= center_x <= 0.65:
                    matched.append(track)
        candidates = matched or tracks
        candidates.sort(key=lambda item: (item.distance_m, -item.last_interaction_ts))
        active = candidates[0]
        reason = "direction_match" if matched else "nearest_or_recent"
        if self._last_active_person_id and self._last_active_person_id != active.person_id:
            reason = f"switch_from_{self._last_active_person_id}_to_{active.person_id}:{reason}"
        self._last_active_person_id = active.person_id
        return TargetSelection(active_person_id=active.person_id, reason=reason)
