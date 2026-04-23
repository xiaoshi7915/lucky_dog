"""个性化语气适配器：根据用户年龄、情绪状态、熟悉程度动态调整机器狗的说话风格。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToneProfile:
    """语气风格描述。"""

    style: str            # 语气名称：gentle / lively / humorous / calm / empathetic / formal
    speed_hint: str       # 语速建议：slow / normal / fast
    formality: str        # 正式程度：casual / semi-formal / formal
    emoji_ok: bool        # 是否可以加入轻量语气词/感叹词
    max_sentence_len: int # 建议最长单句字数（控制语言复杂度）
    instructions: str     # 注入 prompt 的语气指引文字


# 预定义的语气档案
_TONE_PROFILES: dict[str, ToneProfile] = {
    "gentle": ToneProfile(
        style="gentle",
        speed_hint="slow",
        formality="semi-formal",
        emoji_ok=True,
        max_sentence_len=25,
        instructions="用轻柔、温和的语气，语速适当放慢，多停顿，给对方充分空间。避免急迫感。",
    ),
    "lively": ToneProfile(
        style="lively",
        speed_hint="fast",
        formality="casual",
        emoji_ok=True,
        max_sentence_len=20,
        instructions="语气活泼、有感染力，多用感叹词，节奏明快，带动氛围。",
    ),
    "humorous": ToneProfile(
        style="humorous",
        speed_hint="normal",
        formality="casual",
        emoji_ok=True,
        max_sentence_len=22,
        instructions="可以适当接梗、幽默，但不冒犯。用轻松调侃化解尴尬，保持友好底色。",
    ),
    "calm": ToneProfile(
        style="calm",
        speed_hint="normal",
        formality="semi-formal",
        emoji_ok=False,
        max_sentence_len=28,
        instructions="语气平稳、沉着，给人安全感和确定性。不要太多感叹词，用词简洁清晰。",
    ),
    "empathetic": ToneProfile(
        style="empathetic",
        speed_hint="slow",
        formality="casual",
        emoji_ok=True,
        max_sentence_len=20,
        instructions="深度共情，先听后说，用'我理解''我听到了'类语言接住情绪，再温和引导。",
    ),
    "natural": ToneProfile(
        style="natural",
        speed_hint="normal",
        formality="casual",
        emoji_ok=True,
        max_sentence_len=24,
        instructions="自然、亲切，像朋友聊天一样，不做作，适当好奇和提问。",
    ),
    "formal": ToneProfile(
        style="formal",
        speed_hint="normal",
        formality="formal",
        emoji_ok=False,
        max_sentence_len=35,
        instructions="语气正式、礼貌，适合初次正式场合，措辞严谨但不冷漠。",
    ),
}


class PersonalityAdaptor:
    """根据用户年龄、情绪、熟悉程度动态选择最合适的语气风格。"""

    def adapt(
        self,
        emotion_state: str,
        age: int = 0,
        turn_count: int = 0,
        scene_tone: str = "natural",
    ) -> ToneProfile:
        """综合多维度信息选择语气档案。

        优先级：情绪紧急状态 > 场景推荐 > 年龄适配 > 默认自然
        """
        # 1. 情绪紧急状态：直接覆盖场景推荐
        if emotion_state in ("sad", "anxious"):
            return _TONE_PROFILES["empathetic"]
        if emotion_state == "angry":
            return _TONE_PROFILES["calm"]  # 对方愤怒时要冷静应对

        # 2. 场景推荐语气有效时使用
        if scene_tone in _TONE_PROFILES:
            profile = _TONE_PROFILES[scene_tone]
            # 年龄修正：老年人（age > 60）降低活泼度
            if age > 60 and profile.style == "lively":
                return _TONE_PROFILES["natural"]
            # 儿童（age < 10）提升活泼度
            if 0 < age < 10:
                return _TONE_PROFILES["lively"]
            return profile

        # 3. 纯年龄适配（场景未指定时）
        if 0 < age < 10:
            return _TONE_PROFILES["lively"]
        if age > 65:
            return _TONE_PROFILES["calm"]

        # 4. 熟悉程度：陌生人用 natural，老朋友用 lively
        if turn_count == 0:
            return _TONE_PROFILES["natural"]
        if turn_count >= 5:
            return _TONE_PROFILES["lively"]

        return _TONE_PROFILES["natural"]

    def get_reply_style_instruction(
        self,
        emotion_state: str,
        age: int = 0,
        turn_count: int = 0,
        scene_tone: str = "natural",
    ) -> str:
        """返回可直接注入 prompt 的语气指引字符串。"""
        profile = self.adapt(emotion_state, age, turn_count, scene_tone)
        return (
            f"【回复语气要求】风格={profile.style}，"
            f"语速={profile.speed_hint}，"
            f"正式程度={profile.formality}，"
            f"单句建议不超过{profile.max_sentence_len}字。"
            f"{profile.instructions}"
        )
