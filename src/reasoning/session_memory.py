"""升级版会话记忆模块：支持姓名自动提取、情绪趋势历史、话题偏好学习与里程碑记录。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class EmotionSnapshot:
    """单次情绪快照，用于趋势分析。"""

    state: str  # happy / sad / frustrated / neutral 等
    confidence: float
    timestamp: float = field(default_factory=time)


@dataclass
class PersonMemory:
    """单人完整记忆结构（短期 + 中期）。"""

    person_id: str
    preferred_name: str = ""           # 用户自报或自动提取的称呼
    recent_emotion: str = "neutral"    # 最近一次情绪
    emotion_history: list[EmotionSnapshot] = field(default_factory=list)  # 最近 10 次情绪快照
    recent_topics: list[str] = field(default_factory=list)    # 最近 5 个话题
    recent_turns: list[str] = field(default_factory=list)     # 最近 5 轮对话原文
    topic_freq: dict[str, int] = field(default_factory=dict)  # 话题频率（偏好学习）
    turn_count: int = 0                # 累计对话轮次（判断是否熟人）
    first_seen_at: float = field(default_factory=time)        # 首次见面时间
    updated_at: float = field(default_factory=time)           # 最后活跃时间
    milestones: list[str] = field(default_factory=list)       # 重要事件记录（"第一次说名字"等）


# 姓名识别正则：覆盖常见自我介绍句式
_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"我(?:叫|是|的名字(?:是|叫))\s*([^\s，。,!！?？就好吧哦啊啦呀]{1,4})"),
    # "叫我XX" 中排除常见语气词结尾，限制 1~4 字
    re.compile(r"叫我\s*([^\s，。,!！?？就好吧哦啊啦呀]{1,4})"),
    re.compile(r"^([^\s，。,!！?？]{1,4})(?:来了|在的|到了)$"),
]

# 高价值话题关键词（帮助提炼 topic 摘要）
_TOPIC_KEYWORDS: list[str] = [
    "工作", "学习", "考试", "项目", "旅游", "出差", "朋友", "家人", "恋爱",
    "运动", "健身", "游戏", "电影", "音乐", "追剧", "美食", "购物", "宠物",
    "考研", "面试", "毕业", "升职", "失业", "分手", "结婚", "生病",
]


class PersonaSessionMemory:
    """按 person_id 维护多维度视觉语音联合记忆。

    新增能力：
    - 自动从 ASR 文本提取姓名（自我介绍句式识别）
    - 情绪历史快照（支持趋势分析）
    - 话题频率统计（偏好学习）
    - 里程碑记录（第一次见面、第一次说名字等）
    - 可序列化为 prompt 友好摘要
    """

    def __init__(self, ttl_sec: int = 900, max_people: int = 20) -> None:
        self.ttl_sec = ttl_sec
        self.max_people = max_people
        self._store: dict[str, PersonMemory] = {}

    # ------------------------------------------------------------------
    # 主更新接口
    # ------------------------------------------------------------------

    def update(
        self,
        person_id: str,
        topic: str,
        emotion: str,
        utterance: str,
        preferred_name: str = "",
        emotion_confidence: float = 0.0,
    ) -> None:
        """更新某个目标人物的全量记忆。"""
        if not person_id:
            return
        self._evict_expired()
        memory = self._store.get(person_id) or PersonMemory(person_id=person_id)

        # --- 姓名提取（优先显式传入，其次从对话文本自动提取）---
        extracted_name = self._extract_name(utterance)
        name_to_set = preferred_name or extracted_name
        if name_to_set and not memory.preferred_name:
            memory.preferred_name = name_to_set
            memory.milestones.append(f"首次获取姓名={name_to_set} at turn={memory.turn_count}")

        # --- 情绪更新 ---
        if emotion:
            memory.recent_emotion = emotion
            memory.emotion_history = (
                memory.emotion_history + [EmotionSnapshot(state=emotion, confidence=emotion_confidence)]
            )[-10:]  # 保留最近 10 次

        # --- 话题更新 ---
        topic_text = self._extract_topic(topic or utterance)
        if topic_text:
            memory.recent_topics = (memory.recent_topics + [topic_text])[-5:]
            memory.topic_freq[topic_text] = memory.topic_freq.get(topic_text, 0) + 1

        # --- 对话轮次 ---
        if utterance:
            memory.recent_turns = (memory.recent_turns + [utterance[:64]])[-5:]
        memory.turn_count += 1

        # --- 首次见面里程碑 ---
        if memory.turn_count == 1:
            memory.milestones.append("首次见面")

        memory.updated_at = time()
        self._store[person_id] = memory
        self._evict_overflow()

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get(self, person_id: str) -> PersonMemory | None:
        """获取原始记忆对象（用于 ScenarioMatcher 等模块直接访问）。"""
        self._evict_expired()
        return self._store.get(person_id)

    def is_first_meet(self, person_id: str) -> bool:
        """判断是否为首次见面（无记忆或首轮）。"""
        memory = self._store.get(person_id)
        return memory is None or memory.turn_count == 0

    def get_emotion_trend(self, person_id: str) -> str:
        """分析最近 3 次情绪快照的趋势。"""
        memory = self._store.get(person_id)
        if not memory or not memory.emotion_history:
            return "neutral"
        recent = memory.emotion_history[-3:]
        weighted: dict[str, float] = {}
        for i, snap in enumerate(recent):
            w = float(i + 1)
            weighted[snap.state] = weighted.get(snap.state, 0.0) + w * snap.confidence
        return max(weighted, key=lambda e: weighted[e]) if weighted else "neutral"

    def get_favorite_topic(self, person_id: str) -> str:
        """返回最常提及的话题（偏好学习结果）。"""
        memory = self._store.get(person_id)
        if not memory or not memory.topic_freq:
            return ""
        return max(memory.topic_freq, key=lambda t: memory.topic_freq[t])

    def summarize(self, person_id: str) -> str:
        """生成可注入 prompt 的高信息密度记忆摘要。"""
        self._evict_expired()
        memory = self._store.get(person_id)
        if memory is None:
            return "当前目标人物暂无历史记忆。"
        fav = self.get_favorite_topic(person_id)
        trend = self.get_emotion_trend(person_id)
        return (
            f"人物ID={person_id}，"
            f"称呼={memory.preferred_name or '未知'}，"
            f"累计对话={memory.turn_count}轮，"
            f"最近情绪={memory.recent_emotion}（趋势={trend}），"
            f"最近话题={','.join(memory.recent_topics) or '无'}，"
            f"最喜欢聊={fav or '未知'}，"
            f"最近3句={memory.recent_turns[-3:] or ['无']}"
        )

    def summarize_for_scene(self, person_id: str) -> dict[str, Any]:
        """返回结构化摘要供 ScenarioMatcher 使用。"""
        self._evict_expired()
        memory = self._store.get(person_id)
        if memory is None:
            return {
                "preferred_name": "",
                "recent_topics": [],
                "emotion_trend": "neutral",
                "turn_count": 0,
                "is_first_meet": True,
            }
        return {
            "preferred_name": memory.preferred_name,
            "recent_topics": memory.recent_topics,
            "emotion_trend": self.get_emotion_trend(person_id),
            "turn_count": memory.turn_count,
            "is_first_meet": memory.turn_count <= 1,
            "favorite_topic": self.get_favorite_topic(person_id),
            "milestones": memory.milestones[-3:],
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_name(text: str) -> str:
        """从 ASR 文本中自动提取用户姓名。"""
        for pattern in _NAME_PATTERNS:
            m = pattern.search(text)
            if m:
                name = m.group(1).strip()
                # 过滤不合理的"名字"（太短或是常见停用词）
                stopwords = {"你", "我", "他", "她", "不", "没", "是", "了", "吧", "啊"}
                if len(name) >= 1 and name not in stopwords:
                    return name
        return ""

    @staticmethod
    def _extract_topic(text: str) -> str:
        """从文本中提炼话题关键词（命中则返回关键词，否则取前 16 字）。"""
        for kw in _TOPIC_KEYWORDS:
            if kw in text:
                return kw
        cleaned = text.strip()[:16]
        return cleaned if len(cleaned) >= 4 else ""

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
