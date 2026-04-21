"""运行 FunASR 真实麦克风延迟基准。"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import load_config
from src.perception.asr_stream import ASRStream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ASR 真实流式基准")
    parser.add_argument("--duration-sec", type=float, default=30.0, help="采样时长（秒）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config("configs/app.yaml")
    stream = ASRStream(
        model_name=config.asr.model_name,
        sample_rate=config.asr.sample_rate,
        chunk_ms=config.asr.chunk_ms,
        vad_threshold=config.asr.vad_threshold,
    )
    metrics = stream.run_live_stream(duration_sec=args.duration_sec)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
