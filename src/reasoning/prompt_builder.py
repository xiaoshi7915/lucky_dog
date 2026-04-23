"""多模态高情商提示词构造模块：融合情绪、场景、个性化语气层。"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.reasoning.persona_templates import (
    EMOTION_PROMPTS,
    RESEARCH_DISCLAIMER,
    SAFETY_BOUNDARY,
    SCENE_PROMPTS,
    SYSTEM_PERSONA,
    UNIVERSAL_PROHIBITIONS,
)


@dataclass
class PromptContext:
    """高情商提示词上下文结构（升级版）。"""

    asr_text: str
    vision_summary: str
    memory_summary: str = ""

    # 情绪感知层（来自 EmotionAnalyzer）
    emotion_state: str = "neutral"
    emotion_confidence: float = 0.0

    # 场景层（来自 ScenarioMatcher）
    scene_id: str = "s0_default"
    scene_template: str = ""    # 场景话术模板（已填充占位符）
    scene_reply_hint: str = ""  # 场景回复方向提示

    # 个性化层
    preferred_name: str = ""    # 用户称呼，空则"朋友"
    tone_instruction: str = ""  # 语气指引（来自 PersonalityAdaptor）

    # 主动开话层（来自 ProactiveEngagement）
    proactive_hint: str = ""    # 主动开话建议（可空）

    include_disclaimer: bool = True
    recent_topics: list[str] = field(default_factory=list)


class PromptBuilder:
    """高情商提示词构建器：8 层结构化 prompt 组装。"""

    def build(self, context: PromptContext) -> str:
        """按固定层次组装高情商 prompt。

        层次：人设 → 安全边界 → 情绪感知 → 场景指引 → 个性化语气
              → 当前输入 → 记忆摘要 → 主动开话 → 通用禁忌 → 声明
        """
        parts: list[str] = []

        # ── 第1层：核心人设 ──
        parts.append(f"【系统人设】{SYSTEM_PERSONA}")

        # ── 第2层：安全边界 ──
        parts.append(f"【安全边界】{SAFETY_BOUNDARY}")

        # ── 第3层：情绪感知（关键情商层） ──
        emotion_prompt = EMOTION_PROMPTS.get(context.emotion_state, EMOTION_PROMPTS["neutral"])
        confidence_note = ""
        if context.emotion_confidence > 0.5:
            confidence_note = f"（置信度={context.emotion_confidence:.0%}，较高，请认真对待）"
        elif context.emotion_confidence > 0.3:
            confidence_note = f"（置信度={context.emotion_confidence:.0%}，中等，可参考）"
        parts.append(f"【情绪感知】{emotion_prompt}{confidence_note}")

        # ── 第4层：场景指引 ──
        scene_prompt = SCENE_PROMPTS.get(context.scene_id, "")
        if scene_prompt:
            parts.append(f"{scene_prompt}")
        if context.scene_reply_hint:
            parts.append(f"【本轮回复方向】{context.scene_reply_hint}")
        if context.scene_template:
            parts.append(f'【参考开场话术】（可改写，不要照抄）："{context.scene_template}"')

        # ── 第5层：个性化语气 ──
        name_note = f"用户称呼：{context.preferred_name}" if context.preferred_name else "用户称呼：未知（可以先问名字）"
        parts.append(f"【个性化】{name_note}")
        if context.tone_instruction:
            parts.append(f"【语气要求】{context.tone_instruction}")

        # ── 第6层：当前输入 ──
        parts.append(f"【用户说】{context.asr_text or '（用户未发言）'}")
        if context.vision_summary and context.vision_summary != "视觉信息不可用":
            parts.append(f"【视觉感知】{context.vision_summary}")

        # ── 第7层：记忆摘要 ──
        parts.append(f"【历史记忆】{context.memory_summary or '（暂无记忆）'}")
        if context.recent_topics:
            parts.append(f"【近期话题】{', '.join(context.recent_topics[-3:])}")

        # ── 第8层：主动开话建议 ──
        if context.proactive_hint:
            parts.append(f'【主动关怀提示】可以考虑说："{context.proactive_hint}"（根据情况决定是否使用）')

        # ── 通用禁忌 ──
        parts.append(UNIVERSAL_PROHIBITIONS)

        # ── 免责声明 ──
        if context.include_disclaimer:
            parts.append(f"【声明】{RESEARCH_DISCLAIMER}")

        return "\n".join(parts)

    @staticmethod
    def summarize_vision(vision_payload: dict[str, object]) -> str:
        """将结构化视觉结果压缩为可读摘要。"""
        if not vision_payload:
            return "视觉信息不可用"
        gender = vision_payload.get("gender", "unknown")
        age = vision_payload.get("age", "unknown")
        height_cm = vision_payload.get("height_cm", "unknown")
        confidence = vision_payload.get("vision_confidence", "unknown")
        uncertainty_note = vision_payload.get("uncertainty_note", "")
        # face_quality 不再直接输出到 prompt（减少敏感信号，避免模型过度关注颜值）
        gender_text = {"male": "男性", "female": "女性"}.get(str(gender), "性别未知")
        age_text = f"{age}岁左右" if str(age).isdigit() and int(str(age)) > 0 else "年龄未知"
        return (
            f"{gender_text}，{age_text}，估计身高{height_cm}cm，"
            f"视觉置信度={confidence}。{uncertainty_note}"
        )
