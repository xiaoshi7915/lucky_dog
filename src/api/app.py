"""FastAPI 入口。"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.errors import AppError
from src.config import get_config
from src.orchestration.realtime_loop import RealtimeLoop
from src.orchestration.session_manager import SessionManager

logger = logging.getLogger(__name__)

# 全局单例配置，避免多处重复加载 YAML。
config = get_config("configs/app.yaml")

app = FastAPI(title="Lucky Dog Multimodal API", version="0.1.0")

# 全局单例 SessionManager，按 session_id 管理独立 RealtimeLoop。
session_manager = SessionManager(config=config)

# 兼容旧版单轮接口使用的默认 loop（无会话管理场景）。
_default_loop = RealtimeLoop(config=config)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """业务异常处理器，返回结构化错误码（E1000-E4000）。"""
    logger.warning("业务异常 path=%s error_code=%s message=%s", request.url.path, exc.error_code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局兜底异常处理器，统一返回 E5000 系统内部错误格式。"""
    logger.exception("未处理异常 path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error_code": "E5000", "message": "系统内部错误，请稍后重试。", "detail": str(exc)},
    )


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok", "environment": config.environment}


@app.post("/v1/session/start")
def start_session(payload: dict) -> dict:
    """创建新会话并返回 session_id。"""
    import uuid

    session_id = str(payload.get("session_id") or uuid.uuid4())
    mode = str(payload.get("mode", ""))
    entry = session_manager.create_session(session_id=session_id, mode=mode)
    return {"session_id": entry.session_id, "status": entry.status, "mode": entry.mode}


@app.post("/v1/session/{session_id}/turn")
def run_turn(session_id: str, payload: dict) -> dict:
    """单轮对话接口：按 session_id 路由到对应 RealtimeLoop。"""
    # 查找已有会话；未找到时自动创建（兼容直接调用 turn 而不先调用 start 的场景）。
    entry = session_manager.get_session(session_id)
    if entry is None:
        entry = session_manager.create_session(session_id=session_id)
    entry.touch()
    entry.turn_count += 1
    loop = entry.loop

    asr_text = str(payload.get("asrText", ""))
    vision_summary = str(payload.get("visionSummary", ""))
    frame_text = str(payload.get("frameData", ""))
    frame_stream = [frame_text.encode("utf-8")] if frame_text else []
    audio_chunks = [chunk.encode("utf-8") for chunk in asr_text.split() if chunk.strip()]
    if not audio_chunks and asr_text:
        audio_chunks = [asr_text.encode("utf-8")]

    try:
        result = loop.run_one_turn(audio_chunks=audio_chunks, vision_summary=vision_summary, frame_stream=frame_stream)
    except Exception as exc:
        logger.exception("单轮推理失败 session_id=%s", session_id)
        raise HTTPException(status_code=500, detail={"error_code": "E5000", "message": str(exc)}) from exc

    result["session_id"] = session_id
    return result


@app.delete("/v1/session/{session_id}")
def end_session(session_id: str) -> dict:
    """结束并清理指定会话。"""
    ok = session_manager.end_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail={"error_code": "E1000", "message": "会话不存在。"})
    return {"session_id": session_id, "status": "ended"}
