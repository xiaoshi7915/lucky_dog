"""会话记忆模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass
class PersonMemory:
    """单人短期记忆。"""

    person_id: str
    preferred_name: str = ""
    recent_emotion: str = "neutral"
    recent_topics: list[str] = field(default_factory=list)
    recent_turns: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time)


class PersonaSessionMemory:
    """按 person_id 维护视觉语音联合记忆。"""

    def __init__(self, ttl_sec: int = 900, max_people: int = 20) -> None:
        self.ttl_sec = ttl_sec
        self.max_people = max_people
        self._store: dict[str, PersonMemory] = {}

    def update(self, person_id: str, topic: str, emotion: str, utterance: str, preferred_name: str = "") -> None:
        """更新某个目标人物记忆。"""
        if not person_id:
            return
        self._evict_expired()
        memory = self._store.get(person_id) or PersonMemory(person_id=person_id)
        if preferred_name:
            memory.preferred_name = preferred_name
        memory.recent_emotion = emotion or memory.recent_emotion
        if topic:
            memory.recent_topics = (memory.recent_topics + [topic])[-3:]
        if utterance:
            memory.recent_turns = (memory.recent_turns + [utterance])[-3:]
        memory.updated_at = time()
        self._store[person_id] = memory
        self._evict_overflow()

    def summarize(self, person_id: str) -> str:
        """生成可注入提示词的记忆摘要。"""
        self._evict_expired()
        memory = self._store.get(person_id)
        if memory is None:
            return "当前目标人物暂无历史记忆。"
        return (
            f"目标={person_id}，称呼偏好={memory.preferred_name or '未设置'}，"
            f"最近情绪={memory.recent_emotion}，最近主题={','.join(memory.recent_topics) or '无'}，"
            f"最近3轮={memory.recent_turns or ['无']}"
        )

    def _evict_expired(self) -> None:
        now = time()
        expired = [pid for pid, mem in self._store.items() if now - mem.updated_at > self.ttl_sec]
        for pid in expired:
            self._store.pop(pid, None)

    def _evict_overflow(self) -> None:
        if len(self._store) <= self.max_people:
            return
        ordered = sorted(self._store.items(), key=lambda item: item[1].updated_at)
        for pid, _ in ordered[: len(self._store) - self.max_people]:
            self._store.pop(pid, None)
