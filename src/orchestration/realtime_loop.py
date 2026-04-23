"""实时主循环骨架。"""

import logging
from collections.abc import Iterable
from dataclasses import asdict
from queue import Queue
from time import perf_counter
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

from src.actuation.dog_controller_base import DogControllerBase
from src.actuation.dog_controller_mock import DogControllerMock
from src.actuation.dog_controller_unitree import DogControllerUnitree
from src.actuation.tts_engine import TTSEngine
from src.config import AppConfig, load_config
from src.orchestration.event_bus import EventBus
from src.perception.asr_stream import ASRStream
from src.perception.face_analyzer import FaceAnalyzer
from src.perception.multi_person_tracker import MultiPersonTracker
from src.perception.person_face_detector import PersonFaceDetector
from src.perception.pose_height_estimator import PoseHeightEstimator
from src.perception.speaker_direction_estimator import SpeakerDirectionEstimator
from src.perception.target_selector import ActiveSpeakerTargetSelector
from src.reasoning.dialogue_engine import DialogueEngine
from src.reasoning.prompt_builder import PromptBuilder, PromptContext
from src.reasoning.session_memory import PersonaSessionMemory


class RealtimeLoop:
    """编排核心模块形成单轮闭环。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        # 支持外部注入 config（用于 SessionManager 多会话场景），
        # 未注入时回退到从 YAML 加载，保持向后兼容。
        self.config: AppConfig = config if config is not None else load_config("configs/app.yaml")
        self.event_bus = EventBus()
        self.face_analyzer = FaceAnalyzer(self.config.vision)
        self.person_face_detector = PersonFaceDetector(self.config.vision)
        self.height_estimator = PoseHeightEstimator(self.config.vision)
        self.direction_estimator = SpeakerDirectionEstimator()
        self.asr = ASRStream()
        self.prompt_builder = PromptBuilder()
        # 将 LLM 配置注入引擎，确保 model_path / max_new_tokens / temperature 生效。
        self.dialogue = DialogueEngine(
            model_path=self.config.llm.model_path,
            max_new_tokens=self.config.llm.max_new_tokens,
            temperature=self.config.llm.temperature,
        )
        self.tts = TTSEngine(provider=self.config.tts.provider)
        self.tracker = MultiPersonTracker()
        self.target_selector = ActiveSpeakerTargetSelector()
        self.session_memory = PersonaSessionMemory()
        self.dog: DogControllerBase = (
            DogControllerUnitree() if self.config.runtime_mode == "strict_real" else DogControllerMock()
        )
        self.asr_partial_queue: Queue[dict[str, Any]] = Queue(maxsize=self.config.realtime.queue_max_size)
        self.llm_reply_queue: Queue[str] = Queue(maxsize=self.config.realtime.queue_max_size)
        self.tts_play_queue: Queue[bytes] = Queue(maxsize=self.config.realtime.queue_max_size)
        self.state = "LISTENING"
        self._print_readiness_matrix()
        self._ensure_runtime_safety()

    def _print_readiness_matrix(self) -> None:
        """启动时输出真实组件就绪矩阵。"""
        matrix = {
            "ASR": self.config.asr.provider != "mock_asr" and not self.config.asr.use_mock_fallback,
            "LLM": self.config.llm.provider != "mock_llm" and not self.config.llm.use_mock_fallback,
            "TTS": self.config.tts.provider != "mock_tts" and not self.config.tts.use_mock_fallback,
            "Vision": bool(self.config.vision.yolo_model_path and self.config.vision.mmpose_checkpoint_path),
            "Dog": isinstance(self.dog, DogControllerUnitree),
        }
        logger.info("[runtime_mode=%s] readiness=%s", self.config.runtime_mode, matrix)

    def _ensure_runtime_safety(self) -> None:
        """严格模式下执行 fail-fast 就绪检查。"""
        if self.config.runtime_mode != "strict_real":
            return
        readiness = {
            "ASR": self.asr.is_ready,
            "TTS": self.tts.is_ready,
            "Dog": isinstance(self.dog, DogControllerUnitree),
            "MockFallbackForbidden": not any(
                (
                    self.config.asr.use_mock_fallback,
                    self.config.llm.use_mock_fallback,
                    self.config.tts.use_mock_fallback,
                )
            ),
        }
        not_ready = [name for name, ok in readiness.items() if not ok]
        if not_ready:
            raise RuntimeError(f"strict_real 组件未就绪: {','.join(not_ready)}")

    def _build_vision_payload(
        self,
        frame_bytes: bytes,
        frame_index: int = 0,
        speaker_direction: str = "center",
    ) -> dict[str, Any]:
        """从图像帧提取视觉属性结构。"""
        face = self.face_analyzer.analyze(frame_bytes)
        height = self.height_estimator.estimate(frame_bytes)
        vision_confidence = round((face.confidence + height.confidence) / 2, 2)
        low_confidence = vision_confidence < max(
            self.config.vision.thresholds.face_confidence,
            self.config.vision.thresholds.person_confidence,
        )
        uncertainty_note = "视觉置信度不足，以下结论仅供参考。"
        if not low_confidence:
            uncertainty_note = "视觉结论可信度较高。"
        detections = self.person_face_detector.detect(frame_bytes)
        tracks = self.tracker.update(detections=detections, frame_index=frame_index)
        target = self.target_selector.select(tracks=tracks, speaker_direction=speaker_direction)
        return {
            "gender": face.gender if not low_confidence else "unknown",
            "age": face.age if not low_confidence else 0,
            "beauty_score_0_10": face.face_quality,
            "height_cm": height.height_cm,
            "kpt_coverage": height.kpt_coverage,
            "vision_confidence": vision_confidence,
            "uncertainty_note": uncertainty_note,
            "active_person_id": target.active_person_id,
            "target_switch_reason": target.reason,
            "tracked_person_count": len(tracks),
            "speaker_direction": speaker_direction,
        }

    def stream_input_events(
        self,
        frame_stream: Iterable[bytes],
        audio_chunks: Iterable[bytes],
        max_vision_frames: int = 3,
    ) -> dict[str, Any]:
        """持续产生视觉与 ASR 输入事件并返回聚合结果。"""
        audio_chunk_list = list(audio_chunks)
        speaker_direction = self.direction_estimator.estimate(audio_chunk_list, channels=self.asr.channels)
        latest_vision: dict[str, Any] = {}
        vision_frames_processed = 0
        for frame_index, frame_bytes in enumerate(frame_stream):
            if vision_frames_processed >= max_vision_frames:
                break
            latest_vision = self._build_vision_payload(
                frame_bytes,
                frame_index=frame_index,
                speaker_direction=speaker_direction,
            )
            self.event_bus.publish(
                "vision_frame_analyzed",
                {"frame_index": frame_index, "vision": latest_vision},
            )
            vision_frames_processed += 1
        partial_texts: list[str] = []
        for asr_result in self.asr.iter_partial_results(audio_chunk_list):
            partial_texts.append(asr_result.text)
            self.event_bus.publish("asr_partial", asdict(asr_result))
        return {
            "vision": latest_vision,
            "asr_text": " ".join(partial_texts).strip(),
            "vision_frames_processed": vision_frames_processed,
            "speaker_direction": speaker_direction,
        }

    def run_one_turn(
        self,
        audio_chunks: list[bytes],
        vision_summary: str = "",
        frame_stream: Iterable[bytes] | None = None,
    ) -> dict[str, Any]:
        """执行单轮多模态流程并返回观测结果。"""
        turn_start = perf_counter()
        input_start = perf_counter()
        input_state = self.stream_input_events(frame_stream or [], audio_chunks)
        input_latency_ms = round((perf_counter() - input_start) * 1000, 2)
        asr_text = input_state.get("asr_text", "")
        computed_vision = input_state.get("vision", {})
        resolved_vision_summary = vision_summary or self.prompt_builder.summarize_vision(computed_vision)
        llm_start = perf_counter()
        active_person_id = str(computed_vision.get("active_person_id", ""))
        memory_summary = self.session_memory.summarize(active_person_id)
        prompt = self.prompt_builder.build(
            PromptContext(asr_text=asr_text, vision_summary=resolved_vision_summary, memory_summary=memory_summary)
        )
        self.event_bus.publish("prompt_built", {"prompt": prompt})
        dialogue_output = self.dialogue.respond(
            prompt,
            asr_text=asr_text,
            vision_summary=resolved_vision_summary,
            active_person_id=active_person_id,
        )
        self.event_bus.publish("dialogue_generated", asdict(dialogue_output))
        llm_latency_ms = round((perf_counter() - llm_start) * 1000, 2)
        tts_start = perf_counter()
        audio_bytes = self.tts.synthesize(dialogue_output.reply_text)
        self.event_bus.publish("tts_generated", {"audio_size": len(audio_bytes)})
        tts_latency_ms = round((perf_counter() - tts_start) * 1000, 2)
        action_ok = self.dog.execute_action(dialogue_output.action_intent)
        self.session_memory.update(
            person_id=active_person_id,
            topic=asr_text[:32],
            emotion="neutral",
            utterance=asr_text,
        )
        total_latency_ms = round((perf_counter() - turn_start) * 1000, 2)
        result = {
            "prompt": prompt,
            "asr_text": asr_text,
            "vision": computed_vision,
            "vision_frames_processed": input_state.get("vision_frames_processed", 0),
            "dialogue": asdict(dialogue_output),
            "tts_audio_size": len(audio_bytes),
            "action_ok": action_ok,
            "action_history_size": len(self.dog.execution_history),
            "metrics": {
                "degraded": bool(not computed_vision or not asr_text),
                "latency_ms": {
                    "input": input_latency_ms,
                    "dialogue": llm_latency_ms,
                    "tts": tts_latency_ms,
                    "total": total_latency_ms,
                },
                "first_audio_packet_latency_ms": self.tts.first_audio_packet_latency_ms,
            },
        }
        self.event_bus.publish("turn_completed", result)
        return result

    def _drain_queues(self) -> None:
        """清空三个跨轮次队列，防止旧轮次数据在新轮次中积压。"""
        for q in (self.asr_partial_queue, self.llm_reply_queue, self.tts_play_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break

    def run_duplex_turn(
        self,
        audio_chunks: list[bytes],
        frame_stream: Iterable[bytes],
        simulate_barge_in: bool = False,
    ) -> dict[str, Any]:
        """执行全双工并发主循环。"""
        turn_start = perf_counter()
        turn_id = str(uuid4())
        session_id = "local_session"
        # 每轮开始前清空队列，防止前一轮残留数据干扰当前轮次。
        self._drain_queues()
        self.state = "LISTENING"
        input_state = self.stream_input_events(frame_stream=frame_stream, audio_chunks=audio_chunks)
        asr_text = input_state.get("asr_text", "")
        person_id = str(input_state.get("vision", {}).get("active_person_id", ""))
        for partial in self.asr.iter_partial_results(audio_chunks):
            if not self.asr_partial_queue.full():
                self.asr_partial_queue.put(asdict(partial))
                self.event_bus.publish(
                    "partial_text",
                    {"turn_id": turn_id, "session_id": session_id, "person_id": person_id, **asdict(partial)},
                )
        self.state = "THINKING"
        prompt = self.prompt_builder.build(
            PromptContext(
                asr_text=asr_text,
                vision_summary=self.prompt_builder.summarize_vision(input_state.get("vision", {})),
                memory_summary=self.session_memory.summarize(person_id),
            )
        )
        first_token_start = perf_counter()
        # 一次推理同时拿到 DialogueOutput（含动作意图）和 token 迭代器，
        # 避免原来 stream_respond + respond 两次调用造成的重复模型推理。
        dialogue_output, token_iter = self.dialogue.respond_and_stream(
            prompt=prompt,
            asr_text=asr_text,
            vision_summary="",
            active_person_id=str(input_state.get("vision", {}).get("active_person_id", "")),
        )
        tokens: list[str] = []
        for token in token_iter:
            tokens.append(token)
            if not self.llm_reply_queue.full():
                self.llm_reply_queue.put(token)
                self.event_bus.publish(
                    "llm_reply",
                    {"turn_id": turn_id, "session_id": session_id, "person_id": person_id, "token": token},
                )
        first_token_latency_ms = round((perf_counter() - first_token_start) * 1000, 2)
        self.state = "SPEAKING"
        first_audio_start = perf_counter()
        reply_text = "".join(tokens)
        for audio_chunk in self.tts.stream_synthesize([reply_text]):
            if not self.tts_play_queue.full():
                self.tts_play_queue.put(audio_chunk)
                self.event_bus.publish(
                    "tts_packet",
                    {
                        "turn_id": turn_id,
                        "session_id": session_id,
                        "person_id": person_id,
                        "size": len(audio_chunk),
                    },
                )
        first_audio_latency_ms = round((perf_counter() - first_audio_start) * 1000, 2)
        if simulate_barge_in and self.config.realtime.barge_in_enabled:
            self.tts.interrupt_playback()
            self.state = "INTERRUPTED"
        # 直接使用已有的 dialogue_output，无需第二次 respond()。
        action_ok = self.dog.execute_action(dialogue_output.action_intent)
        self.event_bus.publish(
            "motion_cmd",
            {
                "turn_id": turn_id,
                "session_id": session_id,
                "person_id": person_id,
                "action_intent": dialogue_output.action_intent,
                "ok": action_ok,
            },
        )
        return {
            "state": self.state,
            "action_ok": action_ok,
            "turn_id": turn_id,
            "session_id": session_id,
            "person_id": person_id,
            # 视觉数据随结果一起返回，避免调用方再次触发感知链路。
            "vision": input_state.get("vision", {}),
            "metrics": {
                "first_token_latency_ms": first_token_latency_ms,
                "first_audio_latency_ms": first_audio_latency_ms,
                "action_ack_latency_ms": getattr(getattr(self.dog, "_client", None), "last_ack", None).latency_ms
                if hasattr(self.dog, "_client")
                else 0.0,
                "total_latency_ms": round((perf_counter() - turn_start) * 1000, 2),
            },
        }
