#!/usr/bin/env bash
set -euo pipefail

# 切换到仓库根目录，避免相对路径问题。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# 注入项目根目录到 Python 模块搜索路径。
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# 默认使用虚拟环境 Python 执行验收脚本。
.venv/bin/python scripts/run_duplex_acceptance.py "$@"
