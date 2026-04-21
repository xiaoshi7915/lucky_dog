# 真实模式运行调优

## 启动顺序
1. 启动 ASR（FunASR）并确认模型可加载。
2. 启动 LLM（Qwen-7B-Chat AWQ）并观察首 token 延迟。
3. 启动 TTS（CosyVoice）并完成一次流式播报冒烟。
4. 启动全双工 `RealtimeLoop`，验证插话中断。
5. 启动 Unitree 控制器（仿真或真机）并验证安全动作。

## 关键指标
- `first_token_latency_ms`：LLM 首 token 延迟，建议 < 1200ms。
- `first_audio_latency_ms`：TTS 首包延迟，建议 < 1000ms。
- `action_ack_latency_ms`：动作下发确认延迟，建议 < 300ms。
- `total_latency_ms`：整轮总延迟，建议中位数 < 2500ms。

## 调优建议
- 显存紧张时：降低 `max_new_tokens` 或切换小模型 AWQ。
- TTS 抖动时：优先使用 `stream_synthesize` 并降低句长。
- 插话不灵敏时：提升 `barge_in_enabled` 并适当增大 VAD 灵敏度。
- 动作抖动时：提高 Unitree 控制器最小动作间隔。

## 常见故障排查
- 模型加载失败：检查模型目录挂载与读权限。
- 首响应过慢：检查是否误回退到 CPU 推理。
- 打断无效：确认 `realtime.barge_in_enabled=true`。
- 动作不执行：查看 `execution_history` 是否触发限流或不支持动作。

## 验收自动化
- 一键命令：`bash scripts/run_duplex_acceptance.sh`
- 五场景验收：`python scripts/run_e2e_acceptance.py`
- ASR 基准：`python scripts/run_asr_benchmark.py --duration-sec 30`
- TTS 基准：`python scripts/run_tts_benchmark.py --count 20`
- 动作冒烟：`python scripts/run_motion_smoke.py`
- 默认规则：
  - 连续运行 `300` 秒（5 分钟）；
  - 至少触发 `1` 次插话中断；
  - `first_response_median_ms`（首响应中位数）不高于 `2500ms`。
- 快速冒烟（本地）：`bash scripts/run_duplex_acceptance.sh --duration-sec 10`
