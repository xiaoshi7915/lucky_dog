"""场景匹配器：根据当前状态（情绪、记忆、视觉、沉默时长）识别对话场景并加载对应话术模板。"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 场景 ID 字符串类型别名
SceneId = str  # 如 "s1_first_meet" | "s2_reunion" | "s3_comfort" 等

_DEFAULT_SCENES_PATH = "configs/eq_scenarios.yaml"


@dataclass
class SceneMatch:
    """场景匹配结果。"""

    scene_id: SceneId
    scene_name: str
    tone: str  # 推荐语气：gentle / lively / humorous / calm / empathetic
    reply_hint: str  # 本轮回复方向提示，注入 prompt
    action_hint: str  # 推荐动作意图
    template: str  # 随机选取的话术模板（含占位符，如 [NAME]）
    silence_trigger: bool = False  # 是否由沉默超时触发


@dataclass
class SceneContext:
    """场景判断所需的上下文信息。"""

    asr_text: str = ""
    emotion_state: str = "neutral"
    emotion_confidence: float = 0.0
    person_id: str = ""
    preferred_name: str = ""
    recent_topics: list[str] = field(default_factory=list)
    is_first_meet: bool = False
    silence_sec: float = 0.0  # 用户沉默时长（秒）
    tracked_person_count: int = 1
    vision_payload: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0  # 当前 person_id 的历史轮次数


class ScenarioMatcher:
    """从 YAML 配置加载场景规则，根据上下文匹配最佳场景。"""

    def __init__(self, scenes_path: str | Path | None = None) -> None:
        self._raw = self._load_yaml(scenes_path or _DEFAULT_SCENES_PATH)
        self._scenes: dict[SceneId, dict[str, Any]] = {
            s["id"]: s for s in self._raw.get("scenes", [])
        }

    def match(self, ctx: SceneContext) -> SceneMatch:
        """按优先级依次检测场景条件，返回首个命中的场景。

        优先级（高→低）：
        道别 > 多人 > 沉默 > 敏感话题 > 情绪安慰 > 情绪激动 > 情绪开心
        > 初次见面 > 熟人再见 > 默认
        （敏感/情绪场景优先于初次见面，避免被冷启动场景覆盖）
        """
        checks = [
            self._check_farewell,
            self._check_multi_person,
            self._check_silence,
            self._check_sensitive,   # 敏感话题优先于初次见面
            self._check_comfort,     # 情绪安慰优先于初次见面
            self._check_excited,
            self._check_happy,
            self._check_first_meet,  # 初次见面降低优先级
            self._check_reunion,
            self._check_default,
        ]
        for check in checks:
            result = check(ctx)
            if result is not None:
                return result
        return self._build_match("s0_default", ctx)

    # ------------------------------------------------------------------
    # 场景检测方法
    # ------------------------------------------------------------------

    def _check_farewell(self, ctx: SceneContext) -> SceneMatch | None:
        farewell_kws = ["再见", "拜拜", "bye", "走了", "先走", "晚安", "回去了"]
        if any(kw in ctx.asr_text.lower() for kw in farewell_kws):
            return self._build_match("s8_farewell", ctx)
        return None

    def _check_multi_person(self, ctx: SceneContext) -> SceneMatch | None:
        if ctx.tracked_person_count >= 2:
            return self._build_match("s7_multi_person", ctx)
        return None

    def _check_silence(self, ctx: SceneContext) -> SceneMatch | None:
        if ctx.silence_sec >= 60:
            return self._build_match("s5_silence_3", ctx, silence_trigger=True)
        if ctx.silence_sec >= 30:
            return self._build_match("s5_silence_2", ctx, silence_trigger=True)
        if ctx.silence_sec >= 8 and not ctx.asr_text.strip():
            return self._build_match("s5_silence_1", ctx, silence_trigger=True)
        return None

    def _check_first_meet(self, ctx: SceneContext) -> SceneMatch | None:
        if ctx.is_first_meet or ctx.turn_count == 0:
            return self._build_match("s1_first_meet", ctx)
        return None

    def _check_reunion(self, ctx: SceneContext) -> SceneMatch | None:
        if ctx.turn_count > 0 and ctx.preferred_name:
            return self._build_match("s2_reunion", ctx)
        return None

    def _check_sensitive(self, ctx: SceneContext) -> SceneMatch | None:
        sensitive_kws = ["颜值", "好看吗", "漂亮吗", "性格怎么样", "我帅吗", "我丑吗"]
        if any(kw in ctx.asr_text for kw in sensitive_kws):
            return self._build_match("s6_sensitive", ctx)
        return None

    def _check_comfort(self, ctx: SceneContext) -> SceneMatch | None:
        if ctx.emotion_state in ("sad", "frustrated", "anxious") and ctx.emotion_confidence > 0.3:
            return self._build_match("s3_comfort", ctx)
        return None

    def _check_excited(self, ctx: SceneContext) -> SceneMatch | None:
        if ctx.emotion_state == "excited" and ctx.emotion_confidence > 0.4:
            return self._build_match("s4_excited", ctx)
        return None

    def _check_happy(self, ctx: SceneContext) -> SceneMatch | None:
        if ctx.emotion_state == "happy" and ctx.emotion_confidence > 0.3:
            return self._build_match("s4_happy", ctx)
        return None

    def _check_default(self, ctx: SceneContext) -> SceneMatch | None:
        return self._build_match("s0_default", ctx)

    # ------------------------------------------------------------------
    # 结果构建
    # ------------------------------------------------------------------

    def _build_match(
        self, scene_id: SceneId, ctx: SceneContext, silence_trigger: bool = False
    ) -> SceneMatch:
        scene_cfg = self._scenes.get(scene_id, {})
        templates: list[str] = scene_cfg.get("templates", ["我在这里陪着你。"])
        raw_template = random.choice(templates)
        # 替换占位符
        filled = self._fill_template(raw_template, ctx)
        return SceneMatch(
            scene_id=scene_id,
            scene_name=scene_cfg.get("name", scene_id),
            tone=scene_cfg.get("tone", "natural"),
            reply_hint=scene_cfg.get("reply_hint", ""),
            action_hint=scene_cfg.get("action_hint", "wag_tail"),
            template=filled,
            silence_trigger=silence_trigger,
        )

    @staticmethod
    def _fill_template(template: str, ctx: SceneContext) -> str:
        """将话术模板中的占位符替换为实际值。"""
        name = ctx.preferred_name or "朋友"
        topic = ctx.recent_topics[-1] if ctx.recent_topics else "之前聊的"
        result = template
        result = result.replace("[NAME]", name)
        result = result.replace("[TOPIC]", topic)
        result = result.replace("[EMOTION_FEEDBACK]", _emotion_to_feedback(ctx.emotion_state))
        return result

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}


def _emotion_to_feedback(emotion: str) -> str:
    """将情绪状态转为视觉反馈描述词。"""
    return {
        "happy": "很开心的样子",
        "excited": "特别兴奋",
        "sad": "有些低落",
        "frustrated": "有点累了",
        "anxious": "有些紧张",
        "angry": "好像有点不开心",
        "neutral": "挺放松的",
    }.get(emotion, "挺好的")
