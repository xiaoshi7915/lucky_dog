"""运行 CosyVoice 合成与播放基准。"""

from __future__ import annotations

import argparse
import json
import sys
from statistics import median
from time import perf_counter

from src.config import load_config
from src.actuation.tts_engine import TTSEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TTS 真实链路基准")
    parser.add_argument("--count", type=int, default=20, help="合成句子数量")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config("configs/app.yaml")
    engine = TTSEngine(provider=config.tts.provider)
    samples = ["你好，我是 Lucky Dog。"] * max(args.count, 1)
    first_packet_latencies: list[float] = []
    synthesis_latencies: list[float] = []
    playable_ok = 0
    for sentence in samples:
        start = perf_counter()
        chunks = list(engine.stream_synthesize([sentence]))
        first_packet_latencies.append(round((perf_counter() - start) * 1000, 2))
        synthesis_latencies.append(engine.last_synthesis_latency_ms)
        if chunks and engine.play_now(chunks[0]):
            playable_ok += 1
    report = {
        "count": len(samples),
        "first_packet_latency_p50_ms": round(median(first_packet_latencies), 2),
        "synthesis_latency_p50_ms": round(median(synthesis_latencies), 2),
        "playable_success_rate": round(playable_ok / len(samples), 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
