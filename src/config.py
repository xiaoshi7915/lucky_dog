"""项目配置加载模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ASRConfig:
    """ASR 配置。"""

    provider: str
    model_name: str
    sample_rate: int
    chunk_ms: int
    vad_threshold: float
    min_silence_ms: int
    use_mock_fallback: bool


@dataclass
class LLMConfig:
    """LLM 配置。"""

    provider: str
    model_path: str
    quant_type: str
    max_new_tokens: int
    temperature: float
    use_mock_fallback: bool


@dataclass
class TTSConfig:
    """TTS 配置。"""

    provider: str
    voice: str
    emotion: str
    sample_rate: int
    cache_dir: str
    use_mock_fallback: bool


@dataclass
class RealtimeConfig:
    """全双工编排配置。"""

    barge_in_enabled: bool
    min_silence_ms: int
    queue_max_size: int


@dataclass
class DeviceConfig:
    """设备配置。"""

    prefer_device: str
    asr_device: str
    llm_device: str
    tts_device: str


@dataclass
class VisionThresholdConfig:
    """视觉阈值配置。"""

    face_confidence: float
    person_confidence: float
    keypoint_confidence: float
    fqa_confidence: float


@dataclass
class CameraCalibrationConfig:
    """相机标定参数。"""

    camera_height_cm: float
    camera_tilt_deg: float
    focal_length_px: float
    principal_point: tuple[float, float]
    distortion_coeffs: list[float]


@dataclass
class VisionConfig:
    """视觉模块配置。"""

    insightface_model_pack: str
    insightface_providers: list[str]
    fqa_model_path: str
    yolo_model_path: str
    mmpose_config_path: str
    mmpose_checkpoint_path: str
    thresholds: VisionThresholdConfig
    calibration: CameraCalibrationConfig


@dataclass
class AppConfig:
    """应用配置数据结构。"""

    environment: str
    runtime_mode: str
    enable_disclaimer: bool
    asr: ASRConfig
    llm: LLMConfig
    tts: TTSConfig
    realtime: RealtimeConfig
    device: DeviceConfig
    vision: VisionConfig

    @property
    def asr_provider(self) -> str:
        """兼容旧字段：ASR provider。"""
        return self.asr.provider

    @property
    def llm_provider(self) -> str:
        """兼容旧字段：LLM provider。"""
        return self.llm.provider

    @property
    def tts_provider(self) -> str:
        """兼容旧字段：TTS provider。"""
        return self.tts.provider


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典，override 优先。"""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 并保证返回字典。"""
    if not path.exists():
        return {}
    raw_text = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw_text) or {}


_config_singleton: "AppConfig | None" = None
_config_singleton_path: str = ""


def get_config(config_path: str | Path = "configs/app.yaml") -> "AppConfig":
    """获取全局单例配置，避免多处重复加载 YAML 文件。

    首次调用时读取并缓存，后续调用直接返回缓存对象。
    如需强制重新加载，请直接调用 load_config()。
    """
    global _config_singleton, _config_singleton_path  # noqa: PLW0603
    path_str = str(config_path)
    if _config_singleton is None or _config_singleton_path != path_str:
        _config_singleton = load_config(config_path)
        _config_singleton_path = path_str
    return _config_singleton


def load_config(config_path: str | Path, models_path: str | Path | None = None) -> AppConfig:
    """从 YAML 文件加载配置。"""
    path = Path(config_path)
    model_path = Path(models_path) if models_path else path.with_name("models.yaml")
    base_data = _read_yaml(path)
    model_data = _read_yaml(model_path)
    vision_path = path.with_name("vision.yaml")
    vision_data = _read_yaml(vision_path)
    raw_data = _deep_merge(base_data, model_data)
    raw_data = _deep_merge(raw_data, vision_data)
    app_data = raw_data.get("app", {})
    asr_data = raw_data.get("asr", {})
    llm_data = raw_data.get("llm", {})
    tts_data = raw_data.get("tts", {})
    realtime_data = raw_data.get("realtime", {})
    device_data = raw_data.get("device", {})
    vision_raw = raw_data.get("vision", {})
    insightface_data = vision_raw.get("insightface", {})
    fqa_data = vision_raw.get("fqa", {})
    yolo_data = vision_raw.get("yolo", {})
    mmpose_data = vision_raw.get("mmpose", {})
    threshold_data = vision_raw.get("thresholds", {})
    calibration_data = vision_raw.get("calibration", {})
    runtime_mode = str(app_data.get("runtime_mode", "dev_fallback"))
    asr_use_mock = bool(asr_data.get("use_mock_fallback", True))
    llm_use_mock = bool(llm_data.get("use_mock_fallback", True))
    tts_use_mock = bool(tts_data.get("use_mock_fallback", True))
    if runtime_mode == "strict_real" and any((asr_use_mock, llm_use_mock, tts_use_mock)):
        raise ValueError("strict_real 模式下禁止启用 use_mock_fallback。")
    return AppConfig(
        environment=app_data.get("environment", "dev"),
        runtime_mode=runtime_mode,
        enable_disclaimer=bool(app_data.get("enable_disclaimer", True)),
        asr=ASRConfig(
            provider=asr_data.get("provider", app_data.get("asr_provider", "mock_asr")),
            model_name=asr_data.get("model_name", "mock-funasr"),
            sample_rate=int(asr_data.get("sample_rate", 16000)),
            chunk_ms=int(asr_data.get("chunk_ms", 200)),
            vad_threshold=float(asr_data.get("vad_threshold", 0.01)),
            min_silence_ms=int(asr_data.get("min_silence_ms", 300)),
            use_mock_fallback=asr_use_mock,
        ),
        llm=LLMConfig(
            provider=llm_data.get("provider", app_data.get("llm_provider", "mock_llm")),
            model_path=llm_data.get("model_path", ""),
            quant_type=llm_data.get("quant_type", "awq"),
            max_new_tokens=int(llm_data.get("max_new_tokens", 256)),
            temperature=float(llm_data.get("temperature", 0.7)),
            use_mock_fallback=llm_use_mock,
        ),
        tts=TTSConfig(
            provider=tts_data.get("provider", app_data.get("tts_provider", "mock_tts")),
            voice=tts_data.get("voice", "default"),
            emotion=tts_data.get("emotion", "friendly"),
            sample_rate=int(tts_data.get("sample_rate", 22050)),
            cache_dir=tts_data.get("cache_dir", ".cache/tts"),
            use_mock_fallback=tts_use_mock,
        ),
        realtime=RealtimeConfig(
            barge_in_enabled=bool(realtime_data.get("barge_in_enabled", True)),
            min_silence_ms=int(realtime_data.get("min_silence_ms", 300)),
            queue_max_size=int(realtime_data.get("queue_max_size", 16)),
        ),
        device=DeviceConfig(
            prefer_device=device_data.get("prefer_device", "auto"),
            asr_device=device_data.get("asr_device", "cpu"),
            llm_device=device_data.get("llm_device", "cuda:0"),
            tts_device=device_data.get("tts_device", "cpu"),
        ),
        vision=VisionConfig(
            insightface_model_pack=insightface_data.get("model_pack", "buffalo_l"),
            insightface_providers=list(insightface_data.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])),
            fqa_model_path=fqa_data.get("model_path", "models/fqa.onnx"),
            yolo_model_path=yolo_data.get("model_path", "models/yolov8n.pt"),
            mmpose_config_path=mmpose_data.get("config_path", "configs/mmpose/rtmpose-s.py"),
            mmpose_checkpoint_path=mmpose_data.get("checkpoint_path", "models/rtmpose-s.pth"),
            thresholds=VisionThresholdConfig(
                face_confidence=float(threshold_data.get("face_confidence", 0.55)),
                person_confidence=float(threshold_data.get("person_confidence", 0.45)),
                keypoint_confidence=float(threshold_data.get("keypoint_confidence", 0.35)),
                fqa_confidence=float(threshold_data.get("fqa_confidence", 0.3)),
            ),
            calibration=CameraCalibrationConfig(
                camera_height_cm=float(calibration_data.get("camera_height_cm", 95.0)),
                camera_tilt_deg=float(calibration_data.get("camera_tilt_deg", -8.0)),
                focal_length_px=float(calibration_data.get("focal_length_px", 930.0)),
                principal_point=tuple(calibration_data.get("principal_point", [640.0, 360.0])),
                distortion_coeffs=list(calibration_data.get("distortion_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])),
            ),
        ),
    )
