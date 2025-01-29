#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

# venvをアクティベート
source "${SCRIPT_DIR}/venv/bin/activate"

# 使用方法を表示する関数
show_usage() {
    echo "Usage: $0 <output_directory> <video_file1> [<video_file2> ...]"
    echo "Example: $0 /path/to/output /path/to/video1.mp4 /path/to/video2.avi /path/to/video3.mov"
}

# ファイルサイズを取得する関数（クロスプラットフォーム対応）
get_file_size() {
    if [ "$(uname)" = "Darwin" ]; then
        # macOS
        stat -c %z -f "$1"
    else
        # Linux and other Unix-like systems
        stat -c %s "$1"
    fi
}

# 引数の数をチェック
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

source "${SCRIPT_DIR}/venv/bin/activate"

# Python スクリプトのフルパス
PYTHON_SCRIPT="${SCRIPT_DIR}/nishino_tracking_2.py"

# 出力ディレクトリを取得し、絶対パスに変換
output_dir=$(realpath "$1")
shift  # 最初の引数（出力ディレクトリ）を削除

# 出力ディレクトリが存在しない場合は作成
mkdir -p "$output_dir"

# ファイルサイズを取得する関数（クロスプラットフォーム対応）
get_file_size() {
    if [ "$(uname)" = "Darwin" ]; then
        # macOS
        stat -f%z "$1"
    else
        # Linux and other Unix-like systems
        stat -c%s "$1"
    fi
}

# 並列処理の関数
process_video() {
    video_file=$(realpath "$1")
    if [ ! -f "$video_file" ]; then
        echo "File not found: $video_file"
        return 1
    fi
    
    base_name=$(basename "$video_file" | sed 's/\.[^.]*$//')
    
    # ビデオファイルのサイズを取得
    file_size=$(get_file_size "$video_file")
    
    echo "Processing: $video_file (Size: $file_size bytes)"
    
    # pv を使用してプログレスバーを表示しながら Python スクリプトを実行
    pv -pterab -s $file_size "$video_file" | \
    python3 "$PYTHON_SCRIPT" "${video_file}" "${output_dir}" 2>&1 | \
    while IFS= read -r line; do
        echo "[$(basename "$video_file")] $line"
    done
    
    exit_code=${PIPESTATUS[1]}
    
    if [ $exit_code -ne 0 ]; then
        echo "Error processing file: $video_file (Exit code: $exit_code)"
        return 1
    else
        echo "Successfully processed: $video_file"
        return 0
    fi
}

# 関数をエクスポート
export -f get_file_size
export -f process_video

# 並列処理の実行
export PYTHON_SCRIPT
export output_dir

parallel --line-buffer --keep-order process_video ::: "$@"

echo "All video files have been processed."