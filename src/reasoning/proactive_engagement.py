"""主动开话引擎：检测沉默、话题断档、情绪低迷等时机，主动发起对话或追问。"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any


@dataclass
class ProactiveSignal:
    """主动开话信号。"""

    should_engage: bool       # 是否建议主动开话
    trigger_type: str         # 触发类型：silence / topic_end / emotion_drop / follow_up / none
    suggested_utterance: str  # 建议说的话（注入到下一轮 prompt 的 hint 中）
    urgency: float            # 紧迫度 0~1（越高越应该立刻说）


class ProactiveEngagement:
    """主动开话引擎：监控多种时机并产出开话建议。

    触发类型优先级（高→低）：
    1. emotion_drop  - 情绪突然下降（需要立刻关怀）
    2. silence       - 用户沉默超时（分三阶段）
    3. follow_up     - 上一轮有未追问的话题钩子
    4. topic_end     - 话题聊完了，需要新话题
    """

    def __init__(
        self,
        silence_threshold_1: float = 8.0,   # 第一次轻唤阈值（秒）
        silence_threshold_2: float = 30.0,  # 第二次换话题阈值
        silence_threshold_3: float = 60.0,  # 第三次礼貌退出阈值
    ) -> None:
        self._silence_1 = silence_threshold_1
        self._silence_2 = silence_threshold_2
        self._silence_3 = silence_threshold_3
        self._last_user_speech_at: float = time()
        self._last_topics_count: int = 0
        self._pending_follow_up: str = ""   # 上一轮触发的追问钩子

    def record_user_speech(self, asr_text: str) -> None:
        """每次收到有效 ASR 文本时调用，重置沉默计时。"""
        if asr_text.strip():
            self._last_user_speech_at = time()

    def set_follow_up_hook(self, hook: str) -> None:
        """设置下一轮的追问钩子（由 DialogueEngine 在生成回复时调用）。"""
        self._pending_follow_up = hook

    def check(
        self,
        asr_text: str,
        emotion_state: str,
        prev_emotion_state: str,
        recent_topics: list[str],
        preferred_name: str = "",
        vision_payload: dict[str, Any] | None = None,
    ) -> ProactiveSignal:
        """综合检测是否应该主动开话，返回 ProactiveSignal。

        Args:
            asr_text:          当前轮 ASR 文本（可为空）
            emotion_state:     当前情绪
            prev_emotion_state: 上一轮情绪（用于检测突降）
            recent_topics:     近期话题列表
            preferred_name:    用户称呼（用于个性化话术）
            vision_payload:    视觉感知结果（可选）
        """
        name = preferred_name or "朋友"
        silence_sec = time() - self._last_user_speech_at

        # --- 检测1：情绪突然下降（最高优先）---
        if self._is_emotion_drop(prev_emotion_state, emotion_state):
            return ProactiveSignal(
                should_engage=True,
                trigger_type="emotion_drop",
                suggested_utterance=f"感觉{name}好像突然有点低落……没事吧？想聊聊吗？",
                urgency=0.9,
            )

        # --- 检测2：沉默超时 ---
        silence_signal = self._check_silence(silence_sec, name)
        if silence_signal:
            return silence_signal

        # --- 检测3：未处理的追问钩子 ---
        if self._pending_follow_up and not asr_text.strip():
            hook = self._pending_follow_up
            self._pending_follow_up = ""
            return ProactiveSignal(
                should_engage=True,
                trigger_type="follow_up",
                suggested_utterance=hook,
                urgency=0.5,
            )

        # --- 检测4：话题自然结束，引入新话题 ---
        topic = recent_topics[-1] if recent_topics else ""
        if self._is_topic_ending(asr_text) and topic:
            new_topic = self._suggest_new_topic(recent_topics)
            return ProactiveSignal(
                should_engage=True,
                trigger_type="topic_end",
                suggested_utterance=f"刚才聊到{topic}，顺便想问问，{new_topic}",
                urgency=0.3,
            )

        return ProactiveSignal(
            should_engage=False,
            trigger_type="none",
            suggested_utterance="",
            urgency=0.0,
        )

    # ------------------------------------------------------------------
    # 内部检测方法
    # ------------------------------------------------------------------

    def _check_silence(self, silence_sec: float, name: str) -> ProactiveSignal | None:
        if silence_sec >= self._silence_3:
            return ProactiveSignal(
                should_engage=True,
                trigger_type="silence",
                suggested_utterance=f"好吧，{name}先忙，我在旁边陪着，随时叫我哦～",
                urgency=0.2,
            )
        if silence_sec >= self._silence_2:
            return ProactiveSignal(
                should_engage=True,
                trigger_type="silence",
                suggested_utterance=f"要不我给你讲个最近听到的趣事？或者你最近在追什么剧吗？",
                urgency=0.5,
            )
        if silence_sec >= self._silence_1:
            return ProactiveSignal(
                should_engage=True,
                trigger_type="silence",
                suggested_utterance=f"在发呆吗，{name}？要不要聊点什么～",
                urgency=0.6,
            )
        return None

    @staticmethod
    def _is_emotion_drop(prev: str, curr: str) -> bool:
        """判断情绪是否发生明显下降。"""
        positive = {"happy", "excited"}
        negative = {"sad", "frustrated", "anxious", "angry"}
        return prev in positive and curr in negative

    @staticmethod
    def _is_topic_ending(asr_text: str) -> bool:
        """检测当前话语是否含有话题结束信号。"""
        ending_kws = ["就这样", "差不多", "没了", "就这些", "嗯嗯", "好的好的", "哦哦"]
        return any(kw in asr_text for kw in ending_kws)

    @staticmethod
    def _suggest_new_topic(recent_topics: list[str]) -> str:
        """根据已有话题推荐下一个话题的开场白。"""
        topic_transitions: dict[str, str] = {
            "工作": "工作之外，有没有什么特别想做的事？",
            "学习": "学习这么忙，有没有给自己留点放松时间？",
            "旅游": "上次旅游最让你印象深刻的是什么？",
            "朋友": "和朋友最近有聚过吗？",
            "家人": "家里最近都好吗？",
            "游戏": "最近在玩什么游戏，好玩吗？",
        }
        for topic in reversed(recent_topics):
            if topic in topic_transitions:
                return topic_transitions[topic]
        return "你最近有什么新发现或者感兴趣的事吗？"
