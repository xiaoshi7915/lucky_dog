"""多模态提示词构造模块。"""

from dataclasses import dataclass

from src.reasoning.persona_templates import RESEARCH_DISCLAIMER, SAFETY_BOUNDARY, SYSTEM_PERSONA


@dataclass
class PromptContext:
    """提示词上下文结构。"""

    asr_text: str
    vision_summary: str
    memory_summary: str = ""
    include_disclaimer: bool = True


class PromptBuilder:
    """提示词构建器。"""

    def build(self, context: PromptContext) -> str:
        """聚合语音与视觉信息并附加免责声明。"""
        parts = [
            f"系统人设：{SYSTEM_PERSONA}",
            f"安全边界：{SAFETY_BOUNDARY}",
            f"用户语句：{context.asr_text}",
            f"视觉信息：{context.vision_summary}",
            f"人物记忆：{context.memory_summary or '无'}",
        ]
        if context.include_disclaimer:
            parts.append(f"系统声明：{RESEARCH_DISCLAIMER}")
        return "\n".join(parts)

    @staticmethod
    def summarize_vision(vision_payload: dict[str, object]) -> str:
        """将结构化视觉结果压缩为可读摘要。"""
        if not vision_payload:
            return "视觉信息不可用"
        gender = vision_payload.get("gender", "unknown")
        age = vision_payload.get("age", "unknown")
        face_quality = vision_payload.get("beauty_score_0_10", "unknown")
        height_cm = vision_payload.get("height_cm", "unknown")
        confidence = vision_payload.get("vision_confidence", "unknown")
        uncertainty_note = vision_payload.get("uncertainty_note", "")
        return (
            f"性别={gender}，年龄={age}，颜值分={face_quality}/10，估计身高={height_cm}cm，"
            f"视觉置信度={confidence}。{uncertainty_note}"
        )
