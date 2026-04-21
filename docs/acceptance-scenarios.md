# 听说动业务验收场景

## 场景列表

- 场景1（单人近距安静）：验证首字延迟、首包延迟与动作响应稳定。
- 场景2（多人轮流发言）：验证 `active_person_id` 的切换准确性。
- 场景3（背景噪声）：验证 VAD 对空识别率和中断率的影响。
- 场景4（用户插话打断）：验证 `SPEAKING -> INTERRUPTED` 状态转换。
- 场景5（弱光视觉退化）：验证视觉退化时语音链路仍可持续。

## 指标定义

- 目标切换准确率：`target_switch_accuracy`
- 记忆命中率：`memory_hit_rate`
- 误触发动作率：`false_action_trigger_rate`
- 回合延迟：`first_response_median_ms` 与 `metrics.total_latency_ms`
- ASR 首字延迟：`first_token_latency_ms`
- TTS 首包延迟：`first_audio_latency_ms`
- 动作确认时延：`action_ack_latency_ms`
