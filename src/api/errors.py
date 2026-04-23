"""统一错误码定义，与 design.md 规范保持一致。"""

from __future__ import annotations

from fastapi import HTTPException


class AppError(Exception):
    """应用层业务异常基类，携带结构化错误码。"""

    def __init__(self, error_code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> dict[str, str]:
        """转为 JSON 可序列化字典。"""
        return {"error_code": self.error_code, "message": self.message}

    def to_http_exception(self) -> HTTPException:
        """转为 FastAPI HTTPException。"""
        return HTTPException(status_code=self.status_code, detail=self.to_dict())


class ParamError(AppError):
    """E1000：请求参数错误。"""

    def __init__(self, message: str = "请求参数错误。") -> None:
        super().__init__(error_code="E1000", message=message, status_code=400)


class PerceptionError(AppError):
    """E2000：感知模块错误（视觉/ASR）。"""

    def __init__(self, message: str = "感知模块处理失败，请检查输入数据。") -> None:
        super().__init__(error_code="E2000", message=message, status_code=500)


class ReasoningError(AppError):
    """E3000：推理模块错误（LLM/安全护栏）。"""

    def __init__(self, message: str = "推理模块处理失败，系统将使用降级回复。") -> None:
        super().__init__(error_code="E3000", message=message, status_code=500)


class ActuationError(AppError):
    """E4000：执行模块错误（TTS/动作控制）。"""

    def __init__(self, message: str = "执行模块处理失败，请检查设备状态。") -> None:
        super().__init__(error_code="E4000", message=message, status_code=500)


class SystemError(AppError):
    """E5000：系统内部错误。"""

    def __init__(self, message: str = "系统内部错误，请稍后重试。") -> None:
        super().__init__(error_code="E5000", message=message, status_code=500)
