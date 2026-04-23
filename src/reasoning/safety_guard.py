"""安全护栏模块：检测风险表达并替换为安全话术。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SafetyResult:
    """安全审查结果结构。"""

    text: str
    intervened: bool
    matched_rule: str = ""  # 命中的规则名称，便于日志追溯


# 内置默认规则：关键词黑名单（精确匹配）
_DEFAULT_BLOCKED_KEYWORDS: tuple[str, ...] = (
    # 医疗 / 心理专业判断
    "医疗诊断",
    "心理诊断",
    "临床诊断",
    "病情诊断",
    "精神诊断",
    # 绝对化颜值 / 性格结论
    "你一定很丑",
    "你是最丑的",
    "你是最漂亮的",
    "颜值评分绝对",
    # 隐私侵犯
    "你的住址",
    "你的身份证",
    "你的手机号",
    # 歧视性表达
    "残废",
    "智障",
    "神经病",
)

# 内置正则规则：匹配更宽泛的风险模式
_DEFAULT_BLOCKED_PATTERNS: tuple[str, ...] = (
    r"你的性格[是就是一定]+(.*?)[类型型号]",  # 绝对化性格判断
    r"根据.*?(医|心理|精神).*?诊断",          # 伪装成专业诊断
    r"(我|系统).{0,5}(保证|确保|承诺).{0,10}(准确|正确|无误)",  # 过度承诺
)

# 安全替换话术
_SAFE_REPLY = "这个问题我不做专业判断，我们可以聊聊一般建议。"


class SafetyGuard:
    """对话输出安全过滤器，支持关键词黑名单与正则规则双重检测。

    规则来源优先级：外部 YAML 文件 > 内置默认规则。
    YAML 格式示例：
        blocked_keywords:
          - "医疗诊断"
          - "你一定很丑"
        blocked_patterns:
          - "你的性格[是就是]+(.*?)[类型]"
        safe_reply: "这个问题我不做专业判断，我们聊聊一般建议。"
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        rules = self._load_rules(rules_path)
        self._blocked_keywords: tuple[str, ...] = tuple(rules.get("blocked_keywords", _DEFAULT_BLOCKED_KEYWORDS))
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = [
            (raw, re.compile(raw, re.UNICODE))
            for raw in rules.get("blocked_patterns", _DEFAULT_BLOCKED_PATTERNS)
        ]
        self._safe_reply: str = str(rules.get("safe_reply", _SAFE_REPLY))

    @staticmethod
    def _load_rules(rules_path: str | Path | None) -> dict[str, Any]:
        """从 YAML 文件加载自定义规则，文件不存在时返回空字典（使用内置规则）。"""
        if rules_path is None:
            return {}
        path = Path(rules_path)
        if not path.exists():
            return {}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def enforce(self, generated_text: str) -> SafetyResult:
        """检测生成文本是否触发风险规则，命中则替换为安全话术。"""
        # 1. 关键词精确匹配（大小写不敏感）
        lower_text = generated_text.lower()
        for keyword in self._blocked_keywords:
            if keyword.lower() in lower_text:
                return SafetyResult(text=self._safe_reply, intervened=True, matched_rule=f"keyword:{keyword}")

        # 2. 正则模式匹配
        for raw_pattern, compiled in self._compiled_patterns:
            if compiled.search(generated_text):
                return SafetyResult(text=self._safe_reply, intervened=True, matched_rule=f"pattern:{raw_pattern}")

        return SafetyResult(text=generated_text, intervened=False)
