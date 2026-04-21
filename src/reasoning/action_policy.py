"""动作策略模块。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionDecision:
    """动作决策结果。"""

    intent: str
    reason: str


class ActionPolicy:
    """根据回复文本和上下文推断动作意图。"""

    def decide(
        self,
        reply_text: str,
        asr_text: str,
        vision_summary: str,
        active_person_id: str = "",
        emotion_trend: str = "neutral",
        distance_delta: float = 0.0,
    ) -> ActionDecision:
        """按启发式规则输出动作意图。"""
        lowered = f"{reply_text} {asr_text} {vision_summary}".lower()
        if "点头" in lowered:
            return ActionDecision(intent="nod_head", reason="语义中出现点头引导")
        if "尾巴" in lowered or "开心" in lowered:
            return ActionDecision(intent="wag_tail", reason="语义中出现友好互动信号")
        if "过来" in lowered or "靠近" in lowered:
            return ActionDecision(intent="approach", reason="用户或模型表达了靠近语义")
        if emotion_trend == "negative":
            return ActionDecision(intent="nod_head", reason=f"{active_person_id or '目标'}情绪偏负面，优先安抚")
        if distance_delta < -0.2:
            return ActionDecision(intent="step_back", reason=f"{active_person_id or '目标'}接近过快，触发后退限幅")
        if "左" in lowered:
            return ActionDecision(intent="turn_left", reason="语义中出现左转导向")
        if "右" in lowered:
            return ActionDecision(intent="turn_right", reason="语义中出现右转导向")
        if "你好" in lowered or "hello" in lowered:
            return ActionDecision(intent="nod_head", reason="识别到问候语义")
        return ActionDecision(intent="wag_tail", reason="默认友好反馈动作")
