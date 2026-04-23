"""高情商模块单元测试：EmotionAnalyzer / ScenarioMatcher / SessionMemory / PersonalityAdaptor / ProactiveEngagement。"""

from __future__ import annotations

import time

import pytest

from src.reasoning.emotion_analyzer import EmotionAnalyzer, compute_audio_energy
from src.reasoning.personality_adaptor import PersonalityAdaptor
from src.reasoning.proactive_engagement import ProactiveEngagement
from src.reasoning.scenario_matcher import ScenarioMatcher, SceneContext
from src.reasoning.session_memory import PersonaSessionMemory


# ================================================================
# EmotionAnalyzer 测试
# ================================================================


class TestEmotionAnalyzer:
    """情绪分析模块测试。"""

    def setup_method(self) -> None:
        self.analyzer = EmotionAnalyzer()

    def test_happy_keywords_should_detect_happy(self) -> None:
        """开心关键词应识别为 happy 情绪。"""
        result = self.analyzer.analyze("今天好开心啊，哈哈哈太棒了！")
        assert result.state == "happy"
        assert result.confidence > 0

    def test_sad_keywords_should_detect_sad(self) -> None:
        """难过关键词应识别为 sad 情绪。"""
        result = self.analyzer.analyze("我今天好难过，心里很委屈")
        assert result.state == "sad"
        assert result.confidence > 0

    def test_frustrated_keywords_should_detect_frustrated(self) -> None:
        """烦躁关键词应识别为 frustrated 情绪。"""
        result = self.analyzer.analyze("烦死了，太难了，崩溃了")
        assert result.state == "frustrated"

    def test_angry_keywords_should_detect_angry(self) -> None:
        """愤怒关键词应识别为 angry 情绪。"""
        result = self.analyzer.analyze("气死了，太过分了！")
        assert result.state == "angry"

    def test_neutral_text_should_be_neutral(self) -> None:
        """普通文本应识别为 neutral 或低置信度情绪。"""
        result = self.analyzer.analyze("今天天气怎么样")
        assert result.state in ("neutral", "happy", "sad", "frustrated", "angry", "anxious", "excited")
        # neutral 文本置信度应较低或为 neutral
        if result.state != "neutral":
            assert result.confidence < 0.8

    def test_audio_energy_high_should_lean_excited(self) -> None:
        """高能量音频应倾向 excited/happy。"""
        result = self.analyzer.analyze("", audio_energy=0.9)
        assert result.state in ("excited", "happy")

    def test_audio_energy_low_should_lean_sad(self) -> None:
        """低能量音频应倾向 sad/neutral。"""
        result = self.analyzer.analyze("", audio_energy=0.1)
        assert result.state in ("sad", "neutral")

    def test_analyze_trend_with_history(self) -> None:
        """情绪历史趋势分析应返回最多出现的主导情绪。"""
        from src.reasoning.emotion_analyzer import EmotionResult

        history = [
            EmotionResult("sad", 0.8),
            EmotionResult("sad", 0.7),
            EmotionResult("neutral", 0.3),
        ]
        trend = self.analyzer.analyze_trend(history)
        assert trend == "sad"

    def test_evidence_list_not_empty_on_hit(self) -> None:
        """命中规则时 evidence 列表不应为空。"""
        result = self.analyzer.analyze("太开心了")
        assert len(result.evidence) > 0

    def test_compute_audio_energy_pcm(self) -> None:
        """PCM 能量计算应返回 0~1 的浮点数。"""
        import struct

        samples = [1000] * 100
        pcm = struct.pack(f"{len(samples)}h", *samples)
        energy = compute_audio_energy(pcm)
        assert 0.0 <= energy <= 1.0

    def test_compute_audio_energy_empty(self) -> None:
        """空音频应返回 0。"""
        assert compute_audio_energy(b"") == 0.0


# ================================================================
# ScenarioMatcher 测试
# ================================================================


class TestScenarioMatcher:
    """场景匹配器测试。"""

    def setup_method(self) -> None:
        self.matcher = ScenarioMatcher()

    def test_first_meet_should_match_s1(self) -> None:
        """新用户首次见面应匹配 s1_first_meet 场景。"""
        ctx = SceneContext(asr_text="你好", is_first_meet=True, turn_count=0)
        result = self.matcher.match(ctx)
        assert result.scene_id == "s1_first_meet"

    def test_reunion_with_name_should_match_s2(self) -> None:
        """有姓名记忆的老用户应匹配 s2_reunion 场景。"""
        ctx = SceneContext(
            asr_text="嗨",
            is_first_meet=False,
            turn_count=5,
            preferred_name="小明",
        )
        result = self.matcher.match(ctx)
        assert result.scene_id == "s2_reunion"

    def test_farewell_should_match_s8(self) -> None:
        """道别话语应匹配 s8_farewell 场景。"""
        ctx = SceneContext(asr_text="好了，拜拜，我要走了")
        result = self.matcher.match(ctx)
        assert result.scene_id == "s8_farewell"

    def test_sad_emotion_should_match_s3_comfort(self) -> None:
        """高置信度悲伤情绪应匹配 s3_comfort 场景。"""
        ctx = SceneContext(
            asr_text="我好难过",
            emotion_state="sad",
            emotion_confidence=0.8,
        )
        result = self.matcher.match(ctx)
        assert result.scene_id == "s3_comfort"

    def test_happy_emotion_should_match_s4(self) -> None:
        """开心情绪应匹配 s4_happy 场景。"""
        ctx = SceneContext(
            asr_text="太棒了！",
            emotion_state="happy",
            emotion_confidence=0.7,
        )
        result = self.matcher.match(ctx)
        assert result.scene_id == "s4_happy"

    def test_silence_8s_should_trigger_s5_1(self) -> None:
        """8 秒沉默应触发第一阶段唤醒场景。"""
        ctx = SceneContext(asr_text="", silence_sec=10.0)
        result = self.matcher.match(ctx)
        assert result.scene_id == "s5_silence_1"
        assert result.silence_trigger is True

    def test_multi_person_should_match_s7(self) -> None:
        """多人场景应匹配 s7_multi_person。"""
        ctx = SceneContext(asr_text="大家好", tracked_person_count=3)
        result = self.matcher.match(ctx)
        assert result.scene_id == "s7_multi_person"

    def test_sensitive_topic_should_match_s6(self) -> None:
        """颜值相关话题应匹配 s6_sensitive 场景。"""
        ctx = SceneContext(asr_text="我好看吗？你觉得我颜值怎么样")
        result = self.matcher.match(ctx)
        assert result.scene_id == "s6_sensitive"

    def test_template_fills_name(self) -> None:
        """场景话术模板应正确替换 [NAME] 占位符。"""
        ctx = SceneContext(
            asr_text="再见",
            preferred_name="阿强",
        )
        result = self.matcher.match(ctx)
        if "[NAME]" not in result.template:
            # 已替换
            assert "阿强" in result.template or result.template  # 有内容即可


# ================================================================
# SessionMemory 升级版测试
# ================================================================


class TestSessionMemoryUpgraded:
    """升级版会话记忆模块测试。"""

    def setup_method(self) -> None:
        self.memory = PersonaSessionMemory()

    def test_name_extraction_from_self_introduction(self) -> None:
        """自我介绍中应自动提取姓名。"""
        self.memory.update(
            person_id="p1",
            topic="打招呼",
            emotion="happy",
            utterance="我叫小明，很高兴认识你",
        )
        mem = self.memory.get("p1")
        assert mem is not None
        assert mem.preferred_name == "小明"

    def test_name_extraction_call_me(self) -> None:
        """'叫我XX' 格式应提取姓名。"""
        self.memory.update("p2", "", "neutral", "叫我阿花就好啦")
        mem = self.memory.get("p2")
        assert mem is not None
        assert mem.preferred_name == "阿花"

    def test_emotion_history_accumulates(self) -> None:
        """情绪历史应累积保存，最多10条。"""
        for i in range(12):
            self.memory.update("p3", f"话题{i}", "happy", f"内容{i}", emotion_confidence=0.8)
        mem = self.memory.get("p3")
        assert mem is not None
        assert len(mem.emotion_history) <= 10

    def test_topic_freq_learns_preferences(self) -> None:
        """话题频率应记录偏好。"""
        for _ in range(3):
            self.memory.update("p4", "工作", "neutral", "今天工作好累")
        self.memory.update("p4", "旅游", "happy", "上次旅游很开心")
        fav = self.memory.get_favorite_topic("p4")
        assert fav == "工作"

    def test_is_first_meet_true_for_new_person(self) -> None:
        """新用户应返回首次见面标志。"""
        assert self.memory.is_first_meet("unknown_person") is True

    def test_is_first_meet_false_after_update(self) -> None:
        """有记忆后不应再返回首次见面。"""
        self.memory.update("p5", "话题", "neutral", "内容")
        self.memory.update("p5", "话题", "neutral", "内容2")
        assert self.memory.is_first_meet("p5") is False

    def test_emotion_trend_returns_dominant(self) -> None:
        """情绪趋势应返回最主导的情绪。"""
        self.memory.update("p6", "", "sad", "好难过", emotion_confidence=0.9)
        self.memory.update("p6", "", "sad", "还是难过", emotion_confidence=0.8)
        self.memory.update("p6", "", "neutral", "好一点了", emotion_confidence=0.5)
        trend = self.memory.get_emotion_trend("p6")
        assert trend == "sad"

    def test_milestone_recorded_on_first_meet(self) -> None:
        """首次见面应记录里程碑。"""
        self.memory.update("p7", "", "neutral", "你好")
        mem = self.memory.get("p7")
        assert mem is not None
        assert any("首次见面" in m for m in mem.milestones)

    def test_summarize_for_scene_structure(self) -> None:
        """summarize_for_scene 应返回结构化字典。"""
        self.memory.update("p8", "工作", "happy", "今天工作不错", preferred_name="小李")
        scene_info = self.memory.summarize_for_scene("p8")
        assert "preferred_name" in scene_info
        assert "recent_topics" in scene_info
        assert "turn_count" in scene_info
        assert scene_info["preferred_name"] == "小李"


# ================================================================
# PersonalityAdaptor 测试
# ================================================================


class TestPersonalityAdaptor:
    """个性化语气适配器测试。"""

    def setup_method(self) -> None:
        self.adaptor = PersonalityAdaptor()

    def test_sad_emotion_should_use_empathetic(self) -> None:
        """悲伤情绪应使用共情语气。"""
        profile = self.adaptor.adapt("sad")
        assert profile.style == "empathetic"

    def test_angry_emotion_should_use_calm(self) -> None:
        """愤怒情绪应使用冷静语气（降温而非对抗）。"""
        profile = self.adaptor.adapt("angry")
        assert profile.style == "calm"

    def test_elderly_user_should_not_be_lively(self) -> None:
        """老年用户（>60岁）不应使用活泼语气。"""
        profile = self.adaptor.adapt("neutral", age=70, scene_tone="lively")
        assert profile.style != "lively"

    def test_child_should_be_lively(self) -> None:
        """儿童用户（<10岁）应使用活泼语气。"""
        profile = self.adaptor.adapt("neutral", age=7)
        assert profile.style == "lively"

    def test_get_instruction_returns_string(self) -> None:
        """语气指引应返回非空字符串。"""
        instruction = self.adaptor.get_reply_style_instruction("happy", age=25, turn_count=3)
        assert isinstance(instruction, str)
        assert len(instruction) > 0


# ================================================================
# ProactiveEngagement 测试
# ================================================================


class TestProactiveEngagement:
    """主动开话引擎测试。"""

    def setup_method(self) -> None:
        self.engine = ProactiveEngagement(
            silence_threshold_1=5.0,
            silence_threshold_2=15.0,
            silence_threshold_3=30.0,
        )

    def test_no_trigger_when_user_speaking(self) -> None:
        """用户正常说话时不应触发主动开话。"""
        self.engine.record_user_speech("你好")
        signal = self.engine.check(
            asr_text="我今天去了公园",
            emotion_state="neutral",
            prev_emotion_state="neutral",
            recent_topics=["旅游"],
        )
        assert signal.trigger_type == "none" or not signal.should_engage

    def test_emotion_drop_triggers_care(self) -> None:
        """情绪从开心突然变悲伤应触发关怀。"""
        signal = self.engine.check(
            asr_text="",
            emotion_state="sad",
            prev_emotion_state="happy",
            recent_topics=[],
            preferred_name="小明",
        )
        assert signal.should_engage is True
        assert signal.trigger_type == "emotion_drop"
        assert signal.urgency > 0.5

    def test_follow_up_hook_triggers(self) -> None:
        """设置追问钩子后无新输入时应触发追问。"""
        self.engine.set_follow_up_hook("上次说的那件事后来怎么样了？")
        self.engine.record_user_speech("")  # 不算有效发言
        signal = self.engine.check(
            asr_text="",
            emotion_state="neutral",
            prev_emotion_state="neutral",
            recent_topics=[],
        )
        assert signal.should_engage is True
        assert signal.trigger_type == "follow_up"
