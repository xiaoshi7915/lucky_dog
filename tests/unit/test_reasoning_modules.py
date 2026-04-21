"""推理模块单元测试。"""

from src.reasoning.action_policy import ActionPolicy
from src.reasoning.prompt_builder import PromptBuilder, PromptContext
from src.reasoning.safety_guard import SafetyGuard


def test_prompt_builder_should_include_disclaimer_by_default() -> None:
    """默认提示词应附带免责声明。"""
    builder = PromptBuilder()

    prompt = builder.build(PromptContext(asr_text="你好", vision_summary="性别=female"))

    assert "系统声明" in prompt


def test_safety_guard_should_intervene_on_risky_keywords() -> None:
    """命中敏感词时应触发替换。"""
    guard = SafetyGuard()

    result = guard.enforce("这是医疗诊断结论")

    assert result.intervened is True
    assert "不做专业判断" in result.text


def test_action_policy_should_choose_turn_left() -> None:
    """出现左转语义时应输出左转动作。"""
    policy = ActionPolicy()

    decision = policy.decide(reply_text="向左转", asr_text="", vision_summary="")

    assert decision.intent == "turn_left"
