import sys
import os
import argparse
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_video(input_path, output_dir):
    try:
        # 入力ファイル名から拡張子を除いた部分を取得
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # 出力ファイル名を生成（指定されたディレクトリに、同じ名前で_processed.mp4を追加）
        output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
        
        # 動画を読み込む
        clip = VideoFileClip(input_path)
        
        # n フレームごとに1フレームを抽出
        new_clip = clip.subclip().speedx(60)
        
        # 60fpsに設定
        new_clip = new_clip.set_fps(60)
        
        # 処理した動画を保存（常にMP4形式で出力）
        new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4)
        
        # クリップを閉じる
        clip.close()
        new_clip.close()

        return f"Successfully processed {input_path}"
    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Process video files and save them to a specified output directory.")
    parser.add_argument("files", nargs="+", help="Input video files (MP4 or AVI)")
    parser.add_argument("-o", "--output", default=".", help="Output directory for processed videos")
    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(args.output, exist_ok=True)

    # 処理対象のファイルリストを作成
    files_to_process = [f for f in args.files if f.lower().endswith(('.mp4', '.avi'))]

    # 使用可能なCPUスレッド数を取得（最大28）
    max_workers = min(28, len(files_to_process))

    # ThreadPoolExecutorを使用して並列処理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 各ファイルに対してprocess_video関数を実行
        future_to_file = {executor.submit(process_video, file, args.output): file for file in files_to_process}
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'{file} generated an exception: {exc}')

if __name__ == "__main__":
    main()