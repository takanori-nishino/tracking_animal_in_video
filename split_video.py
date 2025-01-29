import cv2
import numpy as np
import moviepy.editor as mp
import ffmpeg
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def split_video(input_file, output_pattern, rows, cols):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_file}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    sub_width = frame_width // cols
    sub_height = frame_height // rows

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_videos = [[cv2.VideoWriter(output_pattern.format(r, c), fourcc, fps, (sub_width, sub_height)) for c in range(cols)] for r in range(rows)]

    for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(input_file)}"):
        ret, frame = cap.read()
        if not ret:
            break
        for r in range(rows):
            for c in range(cols):
                sub_frame = frame[r*sub_height:(r+1)*sub_height, c*sub_width:(c+1)*sub_width]
                out_videos[r][c].write(sub_frame)

    cap.release()
    for row in out_videos:
        for video in row:
            video.release()

def process_video(input_file, output_directory, rows, cols):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_pattern = os.path.join(output_directory, f"{base_name}_r{{}}_c{{}}.avi")
    split_video(input_file, output_pattern, rows, cols)

def main():
    parser = argparse.ArgumentParser(description="Split multiple videos into a grid of smaller videos.")
    parser.add_argument("input_files", nargs="+", help="Input video files (mp4, avi, or wmv)")
    parser.add_argument("-o", "--output_directory", default=".", help="Output directory")
    parser.add_argument("-r", "--rows", type=int, default=2, help="Number of rows in the grid")
    parser.add_argument("-c", "--cols", type=int, default=2, help="Number of columns in the grid")
    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_video, input_file, args.output_directory, args.rows, args.cols) for input_file in args.input_files]
        
        for future in tqdm(as_completed(futures), total=len(args.input_files), desc="Overall Progress"):
            future.result()

if __name__ == "__main__":
    main()