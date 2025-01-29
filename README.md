# Tracking animal in video
This program let you track an animal in video

## Dependencies
This program depends on below.
    Unix 環境
    python 3.10
    cuda-Toolkit
    cuDNN
    opencv_contrib (GPU 対応版)

## Instal
```
    git clone https://github.com/takanori-nishino/tracking_animal_in_video.git
    cd ./tracking_animal_in_video
    python3.10 -m venv venv
    source venv/bin/activate
    sudo apt-get install python3-dev build-essential libyaml-dev # (or, sudo yum install python3-devel libyaml-devel gcc)
    pip install --upgrade pip setuptools wheel
    pip install cython
    pip install PyYAML
    pip install -r requirements.txt
```

## Usage
### Split videos
```
bash split_video.sh [-h] [-o OUTPUT_DIRECTORY] [-r ROWS] [-c COLS] input_files [input_files ...]
```
### Video acceralation
```
bash video_acceralation.sh [-h] [-o OUTPUT] files [files ...]
```
### Nishino tracking xxxx
```
bash nishino_tracking_{xxxx}.sh <output_directory> <video_file1> [<video_file2> ...]
```
### Convert video from xxxx
```
bash convert_video_from_{xxxx}.sh <input_directory> <frame_rate>
```
### Concatenate videos
```
bash concatenate_videos.sh [-h] input_dir output_dir
```

