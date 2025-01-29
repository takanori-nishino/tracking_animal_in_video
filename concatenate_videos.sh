#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

# Pythonスクリプトのパスを設定（必要に応じて変更してください）
PYTHON_SCRIPT="${SCRIPT_DIR}/concatenate_videos.py"

# venvをアクティベート
source "${SCRIPT_DIR}/venv/bin/activate"

# Pythonスクリプトを実行し、すべての引数を渡す
python "$PYTHON_SCRIPT" "$@"

# venvを非アクティベート
deactivate
