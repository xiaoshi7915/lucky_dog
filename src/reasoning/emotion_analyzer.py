"""情绪分析模块：融合文本关键词、语音能量特征与视觉表情信息，输出情绪状态。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 情绪状态枚举字符串，便于类型提示与配置对照
EmotionState = str  # "happy"|"excited"|"sad"|"frustrated"|"anxious"|"angry"|"neutral"

_DEFAULT_RULES_PATH = "configs/eq_scenarios.yaml"


@dataclass
class EmotionResult:
    """情绪分析结果。"""

    state: EmotionState = "neutral"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)  # 命中的证据列表，便于日志追溯


@dataclass
class _EmotionRule:
    """单条文本情绪规则。"""

    emotion: EmotionState
    keywords: list[str]
    patterns: list[str]
    weight: float = 1.0


class EmotionAnalyzer:
    """多信号情绪分析器：文本关键词 + 正则 + 语音能量 + 视觉情绪（可选）。

    设计原则：
    - 任一信号缺失时优雅降级（仅用可用信号）
    - 多信号加权融合，输出置信度
    - 规则来自 eq_scenarios.yaml，支持运行时调整
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        raw = self._load_yaml(rules_path or _DEFAULT_RULES_PATH)
        emotion_rules_raw: list[dict[str, Any]] = raw.get("emotion_text_rules", [])
        self._rules: list[_EmotionRule] = [
            _EmotionRule(
                emotion=r["emotion"],
                keywords=r.get("keywords", []),
                patterns=r.get("patterns", []),
                weight=float(r.get("weight", 1.0)),
            )
            for r in emotion_rules_raw
        ] or self._default_rules()
        # 预编译正则，加速运行时匹配
        self._compiled: list[tuple[_EmotionRule, list[re.Pattern[str]]]] = [
            (rule, [re.compile(p, re.UNICODE | re.IGNORECASE) for p in rule.patterns])
            for rule in self._rules
        ]

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def analyze(
        self,
        asr_text: str,
        audio_energy: float | None = None,
        vision_payload: dict[str, Any] | None = None,
    ) -> EmotionResult:
        """融合文本、语音能量和视觉数据分析情绪。

        Args:
            asr_text:      本轮 ASR 识别文本
            audio_energy:  归一化 PCM 能量值（0~1），可选
            vision_payload: 视觉感知结果字典，可含 face_emotion 字段（可选）

        Returns:
            EmotionResult 带 state / confidence / evidence
        """
        scores: dict[EmotionState, float] = {}
        evidence: list[str] = []

        # --- 文本情绪分析 ---
        text_scores, text_evidence = self._analyze_text(asr_text)
        for emo, score in text_scores.items():
            scores[emo] = scores.get(emo, 0.0) + score * 0.6  # 文本权重 60%
        evidence.extend(text_evidence)

        # --- 语音能量分析 ---
        if audio_energy is not None:
            energy_emo, energy_score, energy_ev = self._analyze_energy(audio_energy)
            scores[energy_emo] = scores.get(energy_emo, 0.0) + energy_score * 0.25  # 能量权重 25%
            evidence.append(energy_ev)

        # --- 视觉情绪分析 ---
        if vision_payload:
            vis_emo, vis_score, vis_ev = self._analyze_vision(vision_payload)
            if vis_score > 0:
                scores[vis_emo] = scores.get(vis_emo, 0.0) + vis_score * 0.15  # 视觉权重 15%
                evidence.append(vis_ev)

        if not scores:
            return EmotionResult(state="neutral", confidence=0.0, evidence=["无有效信号"])

        best_emotion = max(scores, key=lambda e: scores[e])
        total_weight = sum(scores.values())
        confidence = round(min(scores[best_emotion] / max(total_weight, 1e-6), 1.0), 3)

        return EmotionResult(state=best_emotion, confidence=confidence, evidence=evidence)

    def analyze_trend(self, history: list[EmotionResult]) -> EmotionState:
        """从多轮历史中分析情绪趋势（用于 ActionPolicy 的 emotion_trend 参数）。"""
        if not history:
            return "neutral"
        # 取最近 3 轮，加权（越新权重越高：1, 2, 3）
        recent = history[-3:]
        weighted: dict[EmotionState, float] = {}
        for i, result in enumerate(recent):
            w = float(i + 1)
            weighted[result.state] = weighted.get(result.state, 0.0) + w * result.confidence
        return max(weighted, key=lambda e: weighted[e])

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _analyze_text(self, text: str) -> tuple[dict[EmotionState, float], list[str]]:
        """关键词 + 正则多规则情绪打分。"""
        scores: dict[EmotionState, float] = {}
        evidence: list[str] = []
        lower = text.lower()
        for rule, compiled_patterns in self._compiled:
            hit = False
            for kw in rule.keywords:
                if kw in lower:
                    scores[rule.emotion] = scores.get(rule.emotion, 0.0) + rule.weight
                    evidence.append(f"text:keyword:{kw}→{rule.emotion}")
                    hit = True
                    break
            if not hit:
                for pat in compiled_patterns:
                    if pat.search(text):
                        scores[rule.emotion] = scores.get(rule.emotion, 0.0) + rule.weight * 0.8
                        evidence.append(f"text:pattern:{pat.pattern[:20]}→{rule.emotion}")
                        break
        return scores, evidence

    @staticmethod
    def _analyze_energy(energy: float) -> tuple[EmotionState, float, str]:
        """通过语音能量估算情绪激活程度。

        能量越高 → 兴奋/愤怒；能量偏低 → 沮丧/平静。
        """
        if energy > 0.75:
            return "excited", 0.7, f"audio:energy={energy:.2f}→excited"
        if energy > 0.55:
            return "happy", 0.5, f"audio:energy={energy:.2f}→happy"
        if energy < 0.2:
            return "sad", 0.4, f"audio:energy={energy:.2f}→sad"
        return "neutral", 0.3, f"audio:energy={energy:.2f}→neutral"

    @staticmethod
    def _analyze_vision(vision_payload: dict[str, Any]) -> tuple[EmotionState, float, str]:
        """从视觉感知结果提取情绪信号（如 face_emotion 字段或 beauty_score 变化）。"""
        face_emotion = vision_payload.get("face_emotion", "")
        if face_emotion in ("happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"):
            mapping = {
                "happy": "happy",
                "sad": "sad",
                "angry": "angry",
                "surprised": "excited",
                "fearful": "anxious",
                "disgusted": "frustrated",
                "neutral": "neutral",
            }
            mapped = mapping.get(face_emotion, "neutral")
            return mapped, 0.6, f"vision:face_emotion={face_emotion}→{mapped}"
        return "neutral", 0.0, "vision:no_face_emotion"

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

    @staticmethod
    def _default_rules() -> list[_EmotionRule]:
        """内置情绪规则（YAML 缺失时使用）。"""
        return [
            _EmotionRule("happy", ["开心", "高兴", "哈哈", "太好了", "棒", "喜欢", "爱了", "nice", "哇"], [], 1.2),
            _EmotionRule("excited", ["好激动", "太兴奋了", "超期待", "惊喜", "wow", "！！", "厉害了"], [], 1.3),
            _EmotionRule("sad", ["难过", "伤心", "哭了", "委屈", "心疼", "唉", "失落", "遗憾", "可惜"], [], 1.2),
            _EmotionRule(
                "frustrated",
                ["烦死了", "好烦", "崩溃", "受不了", "太难了", "好累", "无语", "气死", "烦透了"],
                [],
                1.2,
            ),
            _EmotionRule("anxious", ["焦虑", "担心", "紧张", "害怕", "怕", "不安", "慌"], [], 1.1),
            _EmotionRule("angry", ["生气", "愤怒", "气死了", "太过分了", "讨厌", "滚"], [], 1.3),
        ]


def compute_audio_energy(pcm_bytes: bytes) -> float:
    """从 PCM16 字节计算归一化 RMS 能量（0~1）。

    可选地传给 EmotionAnalyzer.analyze() 的 audio_energy 参数。
    """
    if not pcm_bytes or len(pcm_bytes) < 2:
        return 0.0
    import struct

    n_samples = len(pcm_bytes) // 2
    try:
        samples = struct.unpack(f"{n_samples}h", pcm_bytes[: n_samples * 2])
        rms = math.sqrt(sum(s * s for s in samples) / n_samples)
        return round(min(rms / 32768.0, 1.0), 4)
    except Exception:
        return 0.0
