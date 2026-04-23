# 机器狗多模态对话系统（MVP）

## 项目目标
本项目用于在 PC 仿真环境落地机器狗多模态实时交互最小可行版本（MVP），打通从摄像头/麦克风输入，到视觉与语音理解、对话生成、语音播报和动作指令输出的完整链路。

## MVP 能力边界
- 输入侧：实时摄像头视频流与麦克风音频流。
- 视觉侧：性别、年龄、人脸质量分（研究实验指标）与身高估计。
- 语音侧：FunASR 流式 ASR、Qwen-7B-Chat AWQ 对话生成、CosyVoice 流式 TTS。
- 动作侧：统一动作接口，支持 mock 与 Unitree 控制器双路径。
- 降级能力：任一模块故障时可降级（例如视觉故障时仅保留语音对话链路）。

## 合规与安全声明
- “颜值/性格”相关结论仅用于娱乐与研究，不构成医学、心理学或任何专业判断。
- 系统输出（文本与语音）必须包含免责声明或等价提示。
- 日志需要记录模型置信度与不确定性，避免将模型输出表达为绝对事实。

## 系统分层
- `perception`：视觉分析、身高估计、流式语音识别。
- `reasoning`：多模态融合、提示词构造、对话与安全护栏。
- `actuation`：语音合成与机器狗动作控制。
- `orchestration`：事件总线、会话管理与实时主循环编排。

## 目录结构
- `README.md`：项目说明、边界与使用方式。
- `requirements.md`：需求定义（用户故事、验收标准、约束、范围边界）。
- `design.md`：系统设计（架构、数据模型、API、错误处理、测试策略）。
- `src/`：业务源码。
- `configs/`：环境、模型、阈值等配置。
- `tests/`：单元、集成、端到端测试目录。
- `.github/workflows/ci.yml`：开源仓库持续集成流水线。
- `Dockerfile` / `docker-compose.yml`：容器化运行配置。
- `CONTRIBUTING.md`：开源协作规范。

## 快速启动（虚拟环境）
1. 创建虚拟环境：`python3 -m venv .venv`
2. 安装开发依赖（含完整 ML 栈）：`.venv/bin/pip install -r requirements-dev.txt`
   - 仅运行测试（无需 GPU）可使用轻量依赖：`.venv/bin/pip install -r requirements-ci.txt`
3. 运行测试（含覆盖率）：`.venv/bin/python -m pytest -q --cov=src`
4. 启动服务：`.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000`

## API 端点
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/v1/session/start` | 创建新会话（返回 session_id） |
| POST | `/v1/session/{session_id}/turn` | 提交单轮多模态输入 |
| DELETE | `/v1/session/{session_id}` | 结束并清理会话 |

## 测试与稳定性
- 单元测试：覆盖 `PromptBuilder`、`SafetyGuard`、`ActionPolicy`、`RealtimeLoop` 等关键逻辑。
- 集成测试：覆盖 FastAPI `/health` 与 `/v1/session/{session_id}/turn`。
- E2E 测试：覆盖单人、多人（高帧输入）和噪声场景。
- 稳定性策略：
  - 视觉处理限流：每轮默认最多处理 3 帧，避免高帧率放大时延。
  - 端到端延迟观测：输出 `metrics.latency_ms`（input/dialogue/tts/total）。
  - 降级可观测：当视觉或语音缺失时返回 `metrics.degraded=true`。
- 全双工观测：输出 `first_token_latency_ms`、`first_audio_latency_ms` 与中断状态机。

## 真实模式说明
- 运行模式：`app.runtime_mode` 支持 `dev_fallback` 与 `strict_real`。当为 `strict_real` 时，`ASR/LLM/TTS` 的 `use_mock_fallback` 必须全部为 `false`，否则启动失败。
- 启动后会打印真实组件就绪矩阵：`ASR/LLM/TTS/Vision/Dog`，用于快速判断是否满足真链路验收条件。
- 全双工状态机：`LISTENING`、`THINKING`、`SPEAKING`、`INTERRUPTED`。
- 打断策略：当检测到插话事件时，TTS 队列立即清空并切换 `INTERRUPTED`。
- Unitree 安全动作：默认启用点头、转向、尾巴动作，并带限流保护。
- 详细调优文档见 `docs/runtime-tuning.md`。
- 视觉真实化：接入 InsightFace（性别/年龄）+ FQA（颜值分）+ YOLOv8/MMPose（身高估计）。
- 标定文档见 `docs/vision-calibration.md`，输出存在误差边界且低置信度会自动降级为不确定话术。

## 一键验收压测
- 目标：自动验证“连续 5 分钟会话 + 至少 1 次插话中断 + 首响应中位数统计”。
- 一键执行：`bash scripts/run_duplex_acceptance.sh`
- 自定义参数示例：`bash scripts/run_duplex_acceptance.sh --duration-sec 300 --min-interruptions 1 --max-first-response-median-ms 2500`
- 输出为 JSON 报告，字段包含：`turn_count`、`interruptions`、`first_response_median_ms`、`passed`。
- 真实摄像头验收还会输出：`vision_detection_rate`、`gender_stability_rate`、`height_jitter_cm`。
- 当 `passed=false` 时脚本返回非 0 退出码，可直接接入 CI 或发布门禁。

## 48小时冲刺新增命令
- ASR 真链路基准：`python scripts/run_asr_benchmark.py --duration-sec 30`
- TTS 真链路基准：`python scripts/run_tts_benchmark.py --count 20`
- Unitree 动作冒烟：`python scripts/run_motion_smoke.py`
- 五场景 E2E 验收：`python scripts/run_e2e_acceptance.py`

## Docker 化运行
1. 构建镜像：`docker compose build`
2. 启动服务：`docker compose up -d`
3. 健康检查：`curl http://localhost:8000/health`

## 开源发布建议
- License：使用 MIT（见 `LICENSE`）。
- 协作规范：见 `CONTRIBUTING.md`。
- CI：GitHub Actions 在 push/PR 自动执行 `pytest -q`。
- 发布前检查：
  - 本地测试全通过；
  - 文档同步更新（`README.md`、`requirements.md`、`design.md`）；
  - 风险声明与免责声明完整可见。

## 里程碑
- M1：文档与代码骨架完成 ✅。
- M2：感知链路打通（视频属性与 ASR）✅。
- M3：对话与播报闭环 ✅。
- M4：动作联动（mock 到真机适配）✅。
- M5：稳定性优化、容错、测试覆盖与发布（**当前阶段**）。
  - [x] 消除 `run_duplex_turn` 重复 LLM 推理
  - [x] `stream_synthesize` 句子级缓冲，降低 TTS 调用次数
  - [x] `SessionManager` 实现多会话隔离
  - [x] `SafetyGuard` 扩展关键词与正则规则，支持 YAML 配置化
  - [x] `ActionPolicy` 规则外化为 `configs/action_rules.yaml`
  - [x] 全局结构化错误码（E1000-E5000）落地
  - [x] CI 轻量化（仅安装 `requirements-ci.txt`）+ 覆盖率报告
  - [x] CosyVoice 引擎单例化，消除每次合成的冷启动延迟

## 风险与缓解
- 算力不足导致时延偏高：采用模型分级、量化与异步流水线。
- 视觉误判导致体验风险：启用安全话术模板与低置信度回退。
- 多模块并发不稳定：使用事件总线、健康检查、重试与熔断机制。
