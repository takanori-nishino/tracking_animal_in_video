#!/bin/bash

# 入力ディレクトリとフレームレートを引数から取得
input_directory="$1"
frame_rate="$2"

# 入力ディレクトリ内のすべての.wmvファイルを処理
for file in "$input_directory"/*.wmv; do
  # 出力ファイル名を生成（.wmvを.aviに置換）
  output_file="${file%.wmv}.avi"
  # ffmpegコマンドを実行して変換
  ffmpeg -hwaccel cuda -i "$file" -r "$frame_rate" -c:v h264_nvenc -c:a aac "$output_file"
done
