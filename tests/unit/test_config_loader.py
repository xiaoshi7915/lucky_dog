"""配置加载模块测试。"""

from __future__ import annotations

from src.config import load_config


def test_load_config_should_include_realtime_and_model_settings() -> None:
    """应加载 ASR/LLM/TTS/全双工和设备配置。"""
    config = load_config("configs/app.yaml", "configs/models.yaml")

    assert config.environment == "dev"
    assert config.runtime_mode in {"dev_fallback", "strict_real"}
    assert config.asr.provider == "funasr_streaming"
    assert config.asr.model_name.startswith("speech_paraformer")
    assert config.llm.quant_type == "awq"
    assert config.tts.provider == "cosyvoice"
    assert config.realtime.barge_in_enabled is True
    assert config.device.llm_device == "cuda:0"
    assert config.vision.insightface_model_pack == "buffalo_l"
    assert config.vision.calibration.focal_length_px > 0


def test_load_config_should_keep_legacy_provider_properties() -> None:
    """应兼容旧 provider 属性访问。"""
    config = load_config("configs/app.yaml", "configs/models.yaml")

    assert config.asr_provider == config.asr.provider
    assert config.llm_provider == config.llm.provider
    assert config.tts_provider == config.tts.provider


def test_load_config_should_reject_mock_fallback_in_strict_mode(tmp_path) -> None:
    cfg = tmp_path / "app.yaml"
    cfg.write_text(
        "\n".join(
            [
                "app:",
                "  runtime_mode: strict_real",
                "asr:",
                "  use_mock_fallback: true",
                "llm:",
                "  use_mock_fallback: false",
                "tts:",
                "  use_mock_fallback: false",
            ]
        ),
        encoding="utf-8",
    )
    try:
        load_config(cfg)
        assert False, "strict_real 应拒绝 mock fallback"
    except ValueError:
        assert True
