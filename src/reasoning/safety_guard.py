"""安全护栏骨架。"""

from dataclasses import dataclass


@dataclass
class SafetyResult:
    """安全审查结果结构。"""

    text: str
    intervened: bool


class SafetyGuard:
    """对话输出安全过滤器。"""

    def enforce(self, generated_text: str) -> SafetyResult:
        """在检测到风险词时替换为中性话术。"""
        blocked_keywords = ("医疗诊断", "心理诊断")
        if any(keyword in generated_text for keyword in blocked_keywords):
            return SafetyResult(text="这个问题我不做专业判断，我们可以聊聊一般建议。", intervened=True)
        return SafetyResult(text=generated_text, intervened=False)
