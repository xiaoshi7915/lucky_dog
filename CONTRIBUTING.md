# 贡献指南

## 本地开发
1. 创建虚拟环境：`python3 -m venv .venv`
2. 安装依赖：`.venv/bin/pip install -r requirements-dev.txt`
3. 运行测试：`.venv/bin/python -m pytest -q`

## 提交流程
1. 新建分支并完成开发。
2. 确保单元、集成、E2E 测试通过。
3. 补充文档（README、requirements、design 中相关部分）。
4. 提交 PR，描述变更动机、测试结果和潜在风险。

## 代码规范
- 优先保持模块边界清晰：`perception` / `reasoning` / `actuation` / `orchestration`。
- 新增能力必须附带测试。
- 涉及“颜值/性格”的输出必须带免责声明。
