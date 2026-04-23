"""对话引擎模块。"""

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

from src.reasoning.action_policy import ActionPolicy
from src.reasoning.persona_templates import RESEARCH_DISCLAIMER
from src.reasoning.safety_guard import SafetyGuard


class _QwenAWQBackend:
    """Qwen AWQ 后端封装，缺依赖时自动回退。"""

    def __init__(self, model_path: str = "", max_new_tokens: int = 256, temperature: float = 0.7) -> None:
        self._pipeline: Any | None = None
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        try:
            from transformers import pipeline  # type: ignore

            # model_path 非空时加载指定本地模型（如 Qwen-7B-Chat AWQ），
            # 否则跳过初始化，等待 _build_reply 降级路径接管。
            if model_path:
                self._pipeline = pipeline("text-generation", model=model_path)
            else:
                self._pipeline = None
        except Exception:
            self._pipeline = None

    def generate(self, prompt: str) -> str:
        """调用模型生成回复。"""
        if self._pipeline is None:
            return ""
        try:
            output = self._pipeline(
                prompt,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
            )
            if isinstance(output, list) and output:
                return str(output[0].get("generated_text", ""))
        except Exception:
            return ""
        return ""


@dataclass
class DialogueOutput:
    """对话结果结构。"""

    reply_text: str
    action_intent: str
    action_reason: str
    safety_intervened: bool


class DialogueEngine:
    """多模态对话引擎实现。"""

    def __init__(self, model_path: str = "", max_new_tokens: int = 256, temperature: float = 0.7) -> None:
        self._guard = SafetyGuard()
        self._policy = ActionPolicy()
        # 将 AppConfig.llm 的参数透传给后端，确保配置生效。
        self._backend = _QwenAWQBackend(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def respond(
        self,
        prompt: str,
        asr_text: str,
        vision_summary: str,
        active_person_id: str = "",
        emotion_trend: str = "neutral",
        distance_delta: float = 0.0,
    ) -> DialogueOutput:
        """生成回复并产出动作意图。"""
        model_reply = self._backend.generate(prompt)
        raw_reply = model_reply.strip() or self._build_reply(asr_text=asr_text, vision_summary=vision_summary)
        safety = self._guard.enforce(raw_reply)
        decision = self._policy.decide(
            reply_text=safety.text,
            asr_text=asr_text,
            vision_summary=vision_summary,
            active_person_id=active_person_id,
            emotion_trend=emotion_trend,
            distance_delta=distance_delta,
        )
        return DialogueOutput(
            reply_text=safety.text,
            action_intent=decision.intent,
            action_reason=decision.reason,
            safety_intervened=safety.intervened,
        )

    def respond_and_stream(
        self,
        prompt: str,
        asr_text: str,
        vision_summary: str,
        active_person_id: str = "",
    ) -> tuple["DialogueOutput", Iterable[str]]:
        """一次推理同时返回完整 DialogueOutput 和 token 迭代器。

        调用方可从 DialogueOutput 获取动作意图，无需二次调用 respond()，
        避免重复模型推理带来的算力浪费。
        """
        output = self.respond(
            prompt=prompt,
            asr_text=asr_text,
            vision_summary=vision_summary,
            active_person_id=active_person_id,
        )

        def _char_iter() -> Iterable[str]:
            yield from output.reply_text

        return output, _char_iter()

    def stream_respond(
        self,
        prompt: str,
        asr_text: str,
        vision_summary: str,
        active_person_id: str = "",
    ) -> Iterable[str]:
        """以流式 token 形式输出回复文本（仅返回 token 流，动作意图请用 respond_and_stream）。"""
        _, token_iter = self.respond_and_stream(
            prompt=prompt,
            asr_text=asr_text,
            vision_summary=vision_summary,
            active_person_id=active_person_id,
        )
        yield from token_iter

    @staticmethod
    def _build_reply(asr_text: str, vision_summary: str) -> str:
        """生成带不确定性约束的话术。"""
        lowered = asr_text.lower()
        sensitive = any(keyword in lowered for keyword in ("颜值", "性格", "好看", "丑", "漂亮"))
        if sensitive:
            return (
                f"{RESEARCH_DISCLAIMER}"
                "我会用轻松方式给你一点反馈，但可能不一定准确。"
                "从你当前表达看，你给人的感觉挺有亲和力。"
            )
        return (
            f"{RESEARCH_DISCLAIMER}"
            f"\n语音摘要：{asr_text or '（未识别到清晰语音）'}"
            f"\n视觉摘要：{vision_summary or '（视觉信息暂不可用）'}"
        )
