# 视觉标定指南

## 目标与误差边界
- 本项目身高估计目标误差为均值接近 ±5cm（单目标定场景）。
- 当光照、遮挡、关键点覆盖度较低时，系统将降级为不确定表达。

## 安装顺序
1. 安装基础依赖：`pip install -r requirements.txt`
2. 安装人脸模型链路：`insightface` + `onnxruntime`
3. 安装人体模型链路：`ultralytics` + `mmpose`

## 标定步骤（已知高度标尺法）
1. 将相机固定在支架上，记录离地高度（`camera_height_cm`）。
2. 测量镜头俯仰角并写入 `camera_tilt_deg`。
3. 在画面中心放置已知身高的人体或标尺，调节 `focal_length_px`。
4. 写入 `principal_point` 与畸变参数 `distortion_coeffs`。
5. 连续采集 30 帧，检查 `height_jitter_cm` 是否可接受。

## 运行与验证
- 验收命令：`python scripts/run_duplex_acceptance.py --duration-sec 60`
- 重点关注字段：
  - `vision_detection_rate`
  - `gender_stability_rate`
  - `height_jitter_cm`
