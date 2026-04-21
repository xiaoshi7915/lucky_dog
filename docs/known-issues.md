# 已知问题清单

- `sounddevice` 或 `simpleaudio` 未安装时，真实 ASR/TTS 链路无法启动。
- `cosyvoice` Python API 在不同版本中方法名可能为 `infer` 或 `inference`，需与当前环境核对。
- Unitree ROS2 真链路目前保留接口，需在现场环境配置 topic 与 QoS 参数。
- 弱光场景下视觉模块可能输出 `unknown`，属于预期降级行为。
