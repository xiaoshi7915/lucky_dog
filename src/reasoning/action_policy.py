"""动作策略模块：从 YAML 配置文件加载触发规则，支持运行时调整。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 默认规则文件路径
_DEFAULT_RULES_PATH = "configs/action_rules.yaml"

# 内置兜底规则（规则文件缺失时使用）
_BUILTIN_KEYWORD_RULES: list[dict[str, Any]] = [
    {"keywords": ["点头"], "intent": "nod_head", "reason": "语义中出现点头引导"},
    {"keywords": ["尾巴", "开心"], "intent": "wag_tail", "reason": "语义中出现友好互动信号"},
    {"keywords": ["过来", "靠近"], "intent": "approach", "reason": "用户或模型表达了靠近语义"},
    {"keywords": ["左"], "intent": "turn_left", "reason": "语义中出现左转导向"},
    {"keywords": ["右"], "intent": "turn_right", "reason": "语义中出现右转导向"},
    {"keywords": ["你好", "hello", "hi"], "intent": "nod_head", "reason": "识别到问候语义"},
]


@dataclass
class ActionDecision:
    """动作决策结果。"""

    intent: str
    reason: str


class ActionPolicy:
    """根据回复文本和上下文推断动作意图，规则来自 YAML 配置文件。

    规则优先级：关键词规则 > 情绪规则 > 距离规则 > 默认规则。
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        rules = self._load_rules(rules_path or _DEFAULT_RULES_PATH)
        self._keyword_rules: list[dict[str, Any]] = rules.get("keyword_rules", _BUILTIN_KEYWORD_RULES)
        self._emotion_rules: list[dict[str, Any]] = rules.get("emotion_rules", [])
        self._scene_action_rules: list[dict[str, Any]] = rules.get("scene_action_rules", [])
        self._distance_rules: list[dict[str, Any]] = rules.get("distance_rules", [])
        self._default_intent: str = str(rules.get("default_intent", "wag_tail"))
        self._default_reason: str = str(rules.get("default_reason", "默认友好反馈动作"))

    @staticmethod
    def _load_rules(rules_path: str | Path) -> dict[str, Any]:
        """从 YAML 文件加载规则，文件不存在时返回空字典（使用内置规则）。"""
        path = Path(rules_path)
        if not path.exists():
            logger.debug("动作规则文件不存在 path=%s，使用内置规则", path)
            return {}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            logger.warning("动作规则文件解析失败 path=%s，使用内置规则", path)
            return {}

    def decide(
        self,
        reply_text: str,
        asr_text: str,
        vision_summary: str,
        active_person_id: str = "",
        emotion_trend: str = "neutral",
        distance_delta: float = 0.0,
        scene_id: str = "",
    ) -> ActionDecision:
        """按 YAML 规则顺序匹配并输出动作意图。

        优先级：关键词 > 场景动作 > 情绪 > 距离 > 默认
        """
        combined = f"{reply_text} {asr_text} {vision_summary}".lower()

        # 1. 关键词规则（顺序匹配，首个命中生效）
        for rule in self._keyword_rules:
            keywords: list[str] = rule.get("keywords", [])
            if any(kw.lower() in combined for kw in keywords):
                return ActionDecision(intent=rule["intent"], reason=rule["reason"])

        # 2. 场景动作映射（场景 ID 精确匹配）
        if scene_id:
            for rule in self._scene_action_rules:
                if rule.get("scene_id", "") == scene_id:
                    return ActionDecision(intent=rule["intent"], reason=rule["reason"])

        # 3. 情绪规则（情绪字符串精确匹配）
        for rule in self._emotion_rules:
            if emotion_trend == rule.get("emotion", ""):
                reason = str(rule.get("reason", "")).replace("目标", active_person_id or "目标")
                return ActionDecision(intent=rule["intent"], reason=reason)

        # 4. 距离规则
        for rule in self._distance_rules:
            threshold = float(rule.get("threshold", 0.0))
            if distance_delta < threshold:
                reason = str(rule.get("reason", "")).replace("目标", active_person_id or "目标")
                return ActionDecision(intent=rule["intent"], reason=reason)

        return ActionDecision(intent=self._default_intent, reason=self._default_reason)
