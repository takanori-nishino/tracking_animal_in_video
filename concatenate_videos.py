import os
import argparse
import subprocess
import json
from datetime import timedelta

def get_creation_time(file_path):
    return os.path.getctime(file_path)

def get_video_info(file_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        file_path
    ]
    output = subprocess.check_output(cmd).decode('utf-8')
    return json.loads(output)

def create_concat_file(video_files, concat_file_path):
    with open(concat_file_path, 'w') as f:
        for video, duration in video_files:
            f.write(f"file '{video}'\n")
            f.write(f"duration {duration}\n")

def concat_videos(input_dir, output_dir):
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.avi', '.mp4', '.png', '.jpg', '.jpeg'))]
    
    if not video_files:
        print(f"No video or image files found in {input_dir}")
        return
    
    video_files.sort(key=lambda x: get_creation_time(os.path.join(input_dir, x)))
    
    processed_files = []
    total_duration = 0
    
    for file in video_files:
        file_path = os.path.join(input_dir, file)
        info = get_video_info(file_path)
        
        if 'streams' in info and len(info['streams']) > 0:
            # Video file
            if 'duration' in info['format']:
                duration = float(info['format']['duration'])
            else:
                # Estimate duration from frames and frame rate
                stream = info['streams'][0]
                if 'nb_frames' in stream and 'avg_frame_rate' in stream:
                    nb_frames = int(stream['nb_frames'])
                    fps = eval(stream['avg_frame_rate'])
                    duration = nb_frames / fps if fps else 0
                else:
                    print(f"Warning: Cannot determine duration for {file}. Skipping.")
                    continue
        else:
            # Assume it's a static image, set duration to 1 second
            duration = 1
        
        processed_files.append((file_path, duration))
        total_duration += duration
    
    if not processed_files:
        print("No valid files to process.")
        return
    
    concat_file_path = os.path.join(output_dir, 'concat_list.txt')
    create_concat_file(processed_files, concat_file_path)
    
    output_file = os.path.join(output_dir, f"concatenated_video_{total_duration:.2f}s.mp4")
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file_path,
        '-vsync', 'vfr',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-an',
        output_file
    ]
    
    print(f"Starting video concatenation. This may take a while...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video concatenation completed. Output saved as {output_file}")
    print(f"Total duration: {timedelta(seconds=total_duration)}")
    
    # Clean up temporary file
    os.remove(concat_file_path)

def main():
    parser = argparse.ArgumentParser(description="Concatenate video files in a directory")
    parser.add_argument("input_dir", help="Input directory containing video files")
    parser.add_argument("output_dir", help="Output directory for the concatenated video")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return
    
    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist")
        return
    
    concat_videos(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()