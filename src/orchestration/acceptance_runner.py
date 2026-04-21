"""全双工验收运行器。"""

from __future__ import annotations

from statistics import median
from time import monotonic, sleep
from typing import Any

from src.orchestration.realtime_loop import RealtimeLoop
from src.perception.camera_input import CameraInput


def run_duplex_acceptance(
    loop: RealtimeLoop,
    camera: CameraInput,
    duration_sec: float = 300.0,
    min_interruptions: int = 1,
    max_first_response_median_ms: float = 2500.0,
    sleep_interval_sec: float = 0.02,
    scenario_name: str = "scenario1_single_user",
) -> dict[str, Any]:
    """运行全双工验收并返回汇总结果。"""
    # 初始化计数器与观测序列。
    turn_count = 0
    interruptions = 0
    # 这里将“首响应”定义为首音频包延迟。
    first_response_samples_ms: list[float] = []
    # 至少安排一次插话中断，默认在第一回合触发。
    forced_interrupt_turn_index = 0
    vision_detected_count = 0
    stable_gender_count = 0
    height_samples: list[float] = []
    target_switches = 0
    memory_hits = 0
    action_false_positives = 0
    previous_active_person_id = ""
    previous_gender = ""
    # 记录开始时间用于控制总时长。
    start_time = monotonic()
    # 循环执行直到达到目标时长。
    while (monotonic() - start_time) < duration_sec:
        # 生成简单音频输入片段，驱动一次回合。
        if scenario_name == "scenario2_multi_person":
            audio_chunks = [b"left speaker", b"right speaker", b"switch turn"]
        elif scenario_name == "scenario3_noise":
            audio_chunks = [b"\x00\x00" * 3200, b"hello", b"\x00\x00" * 1600]
        elif scenario_name == "scenario4_barge_in":
            audio_chunks = [b"please stop", b"barge in now", b"continue"]
        elif scenario_name == "scenario5_low_light":
            audio_chunks = [b"still listening in dark", b"voice only", b"test"]
        else:
            audio_chunks = [b"hello", b"lucky dog", b"test"]
        frame_stream = list(camera.read_frames(max_frames=1))
        if not frame_stream:
            frame_stream = [b"\x00"] if scenario_name == "scenario5_low_light" else [b""]
        # 确保至少一次打断发生。
        simulate_barge_in = turn_count == forced_interrupt_turn_index
        # 调用全双工回合执行。
        result = loop.run_duplex_turn(
            audio_chunks=audio_chunks,
            frame_stream=frame_stream,
            simulate_barge_in=simulate_barge_in,
        )
        # 累计回合数。
        turn_count += 1
        # 采集状态与指标。
        state = str(result.get("state", ""))
        metrics = result.get("metrics", {})
        vision_payload = loop.stream_input_events(frame_stream=frame_stream, audio_chunks=[]).get("vision", {})
        # 统计中断次数。
        if state == "INTERRUPTED":
            interruptions += 1
        # 统计首响应样本。
        first_response_ms = float(metrics.get("first_audio_latency_ms", 0.0))
        first_response_samples_ms.append(first_response_ms)
        active_person_id = str(vision_payload.get("active_person_id", ""))
        if previous_active_person_id and active_person_id and active_person_id != previous_active_person_id:
            target_switches += 1
        previous_active_person_id = active_person_id
        if active_person_id:
            memory_text = loop.session_memory.summarize(active_person_id)
            if "暂无历史记忆" not in memory_text:
                memory_hits += 1
        if not result.get("action_ok", True):
            action_false_positives += 1
        if vision_payload.get("vision_confidence", 0.0) > 0:
            vision_detected_count += 1
        current_gender = str(vision_payload.get("gender", "unknown"))
        if previous_gender and current_gender == previous_gender and current_gender != "unknown":
            stable_gender_count += 1
        previous_gender = current_gender
        height_cm = float(vision_payload.get("height_cm", 0.0))
        if height_cm > 0:
            height_samples.append(height_cm)
        # 轻量 sleep，避免无意义空转。
        sleep(sleep_interval_sec)
    # 计算首响应中位数。
    first_response_median_ms = float(median(first_response_samples_ms)) if first_response_samples_ms else 0.0
    # 统一判定验收通过条件。
    height_jitter_cm = (max(height_samples) - min(height_samples)) if len(height_samples) >= 2 else 0.0
    vision_detection_rate = (vision_detected_count / turn_count) if turn_count else 0.0
    gender_stability_rate = (stable_gender_count / max(turn_count - 1, 1)) if turn_count else 0.0
    passed = (
        turn_count > 0
        and interruptions >= min_interruptions
        and first_response_median_ms <= max_first_response_median_ms
    )
    target_switch_accuracy = target_switches / max(1, turn_count)
    memory_hit_rate = memory_hits / max(1, turn_count)
    false_action_trigger_rate = action_false_positives / max(1, turn_count)
    # 返回结构化报告，供脚本打印和 CI 使用。
    return {
        "duration_sec": duration_sec,
        "turn_count": turn_count,
        "interruptions": interruptions,
        "first_response_median_ms": first_response_median_ms,
        "vision_detection_rate": round(vision_detection_rate, 3),
        "gender_stability_rate": round(gender_stability_rate, 3),
        "height_jitter_cm": round(height_jitter_cm, 2),
        "max_first_response_median_ms": max_first_response_median_ms,
        "min_interruptions": min_interruptions,
        "passed": passed,
        "target_switch_accuracy": round(target_switch_accuracy, 3),
        "memory_hit_rate": round(memory_hit_rate, 3),
        "false_action_trigger_rate": round(false_action_trigger_rate, 3),
    }
