"""FastAPI 入口骨架。"""

from fastapi import FastAPI

from src.config import load_config
from src.orchestration.realtime_loop import RealtimeLoop


app = FastAPI(title="Lucky Dog Multimodal API", version="0.1.0")
loop = RealtimeLoop()
config = load_config("configs/app.yaml")


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok", "environment": config.environment}


@app.post("/v1/session/{session_id}/turn")
def run_turn(session_id: str, payload: dict) -> dict:
    """单轮对话接口占位实现。"""
    _ = session_id
    asr_text = str(payload.get("asrText", ""))
    vision_summary = str(payload.get("visionSummary", ""))
    frame_text = str(payload.get("frameData", ""))
    frame_stream = [frame_text.encode("utf-8")] if frame_text else []
    audio_chunks = [chunk.encode("utf-8") for chunk in asr_text.split() if chunk.strip()]
    if not audio_chunks and asr_text:
        audio_chunks = [asr_text.encode("utf-8")]
    result = loop.run_one_turn(audio_chunks=audio_chunks, vision_summary=vision_summary, frame_stream=frame_stream)
    return result
