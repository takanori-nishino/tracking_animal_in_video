# ライセンスキーを設定
# PySimpleGUI_License = "e0yRJZMmarWyNClMbTnwNLlbVCHPlpweZnSlI76mIBknRflmd9mqVUsGbF33BllpcfiqIIsIIvkwxepgYq2CV9uBcF2gV0JSRdC3Ik66MHTUcoyOMbThMz3TNljBkoxiNOimwEi8TFGflHjYZEWQ5QzrZ8UkRPlgcNG2xLv9ecW31UlabsnKRpW8ZpXXJhzRabWR9kujIAjHoEx6L8CMJlOyYyWV14l5RLmklXyIcc3sQ4iUOaiIJqU8Y6W6t7hDbsme9cyTaASJIAsYIpk85Vhzb1W4VoMhYWX3Nv0mIVjBohiiTqmWlPzKaUGfldutbwyxICsOIXkHNkvibPXVBZhabwnRkUiyOIivIyiFLIC3JPDRdzX4Na00bM221slYcpkwlWECIujYociPM9zSEswSMaTDUoiKLNCYJJEGY5XFRSlcSCX6NpzrddWRVFkSIDjTo8iHMDDicCvbMZTvkzvBM6jPAxyVNhCPIqsiImkzRfhgduGwVmFweKHZBhpKcTmmV2z6IpjWogirMJDlcpvwMPTek9vVMWjgAqy6NiSzIxs4IlkMVstSYnWXlYsYQrWERckGc2mKVrz8chyVIB6EIajEFitFZvDIETwDMvT2El5Qd80yBCnSb8WIFlpXboC85djObU2k0yi9L9C6JDJ1UzEwFKk9ZAHgJ0llcU3IMhi3OxiSIaxwMezqMouwM3yZ4IykM0DSEPuVMpTLEiwcIknr07=m8520baeeaa19dd2c0f317d3ba0648a068ccd2ed614644454bca9a8547fa86e297b1df5ea64904b0f2d2ac3b181f34b186c7eb8b73cd849e43a5d332ff0e85b927707bbbae98098e0d4310081378f5c644f56c1784c78e65bac28da6b33a45bd95011ca2b6c1c90fd833a739c994cf1821f90e175550c327013bcb9661ee5d4bc2e66d3ebe6c5905b8c4914a0fefc3f9b61f36749485056035b757f7b1909a5f1b4f24d015474031c4e445c5044d8cb9fd2c17244d88c31bd1d74c12505de95309a63c3602052fccf468e7b58aad04612938ed60f58bf0fb0c986f1ffd4bd43fcf0fb1b86d3a1e44c8146bf1506207ffa58a21618b2486bcb53225bd92e2d74de1c6b32f7f61a6a3a505a21d62dc9fec6f2e92c010ec1cdc8b73fbf0faf7fc004041ad1f12d566e4dc5b70558cd8e55f23eb1eb1871805c61f7d4dac239f476bf2cfeec7b8efcb9edfb4ac275c40f726a169095e523455f95f0b857d58f5ca8bd04c93a4c9f4e18b79f3c08bc1cc6daf46e989fffe802392521dd98e859f5cb0f8c637a82854b5800031d5b34f5870a4cc0edeb3fe199bc73d7ac7742d79dc65efcf1c4e2ab0a0efb870aed924d9e917fc4430f000b20de8e35166f45536766308427eb4d564e2e0fa7d52c9f6bbff57da4ca21b5004d8f3843402d0cdf2ddddaf78cb84633671be244ba228a9490c394ad32a48533aadb6eec8e2444a282c131"
import sys
import os
import io
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import PySimpleGUI as sg
from PIL import Image
import pandas as pd
from collections import deque
from scipy.optimize import minimize
import time
import concurrent.futures
import queue
from tqdm import tqdm
import traceback
import concurrent.futures
import cupy as cp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# sg.home()

df = pd.DataFrame({'pos_x': [],
                'pos_y': [],
                'is_in_shelter': []},
                index=[])

# 直近の20個のareaデータを保存するリスト
recent_areas = []
# サンプリングするフレーム数を設定
num_samples = 100
target_median = 128  # 目標中央値
target_iqr = 50      # 目標IQR
threshold_ratio = 0.00001 # デフォルトでは 0.01% 領域のみを選択
shelter_pos = None
insect_in_shelter = False
insect_may_in_shelter = [False,False,False]
insect_trail = deque(maxlen=10)  # 昆虫の軌跡を保存するデック
crop_frame = None
frame = None
img = None
img_copy = None
drawing = False
insect_pos = None
prev_insect_pos = None
prev_frame_idx = None
next_insect_pos = None
next_frame_idx = None
interpolate_sw = False
sampled_frames = []
frame_idx = 0
insect_data = []
start_time = time.time()
start_point = None
end_point = None
circle_start = None
circle_end = None
selecting_circle = False
specific_circles = [None] * 9  # 9つの特定領域を格納するリスト
current_mode = 0  # 0: アリーナ, 1-9: 特定領域


##################################################
# ↓ ファイル選択
##################################################

def select_input_files():
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを表示しない
    file_paths = filedialog.askopenfilenames(
        title="Select input video files",
        filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
    )
    return file_paths

def select_output_directory():
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを表示しない
    directory = filedialog.askdirectory(title="Select output directory")
    return directory

##################################################
# ↓ 動画からフレームのサンプリング
##################################################

def sample_frame(args):
    cap, frame_number = args
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return None

def sample_frames_gpu(input_file, num_samples):
    global sampled_frames
    cap = cv2.VideoCapture(input_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampling_interval = total_frames // num_samples
    
    # フレーム番号のリストを生成
    frame_numbers = [i * sampling_interval for i in range(num_samples)]
    
    # マルチスレッディングを使用してフレームを読み込む
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(sample_frame, (cv2.VideoCapture(input_file), frame_number)) for frame_number in frame_numbers]
        for future in concurrent.futures.as_completed(futures):
            frame = future.result()
            if frame is not None:
                sampled_frames.append(frame)
    
    cap.release()
    
    # GPUにデータを転送
    gpu_frames = cp.array(sampled_frames)
    
    # GPUでパーセンタイル計算を行う
    percentiles = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    percentile_frames = [cp.percentile(gpu_frames, p, axis=0).astype(cp.uint8) for p in percentiles]
    
    # 結果をCPUに戻す
    percentile_frames_cpu = [cp.asnumpy(frame) for frame in percentile_frames]
    
    return percentile_frames_cpu, percentiles, total_frames

##################################################
# ↓ マルチスレッド
##################################################

def process_frame(frame, adjusted_background, mask, target_median, target_iqr):
    global threshold_ratio
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adjusted_gray = adjust_brightness(gray, target_median, target_iqr, mask)
        diff = cv2.absdiff(adjusted_gray, adjusted_background)
        denoised = denoise(diff)
        masked_denoised = cv2.bitwise_and(denoised, mask)
        thresh_val = find_threshold(masked_denoised)
        _, masked_thresh = cv2.threshold(masked_denoised, thresh_val, 255, cv2.THRESH_BINARY)
        return adjusted_gray, diff, masked_thresh
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return None, None, None

def run_selector():
    selector = PercentileSelector(percentile_frames, percentiles)
    return selector.run()

##################################################
# ↓ アリーナ認識
##################################################

def enhance_image_visibility(image):
    # グレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # ヒストグラム均一化を適用
    equalized = cv2.equalizeHist(gray)

    # コントラスト制限付き適応ヒストグラム均一化（CLAHE）を適用
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(equalized)

    # カラー画像に戻す
    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_color

# 円を描画するマウスのコールバック関数
def draw_circle(event, x, y, flags, param):
    global start_point, end_point, img, img_copy, current_mode, specific_circles

    if event == cv2.EVENT_LBUTTONDOWN:
        img_copy = img.copy()
        
        if current_mode == 0:  # アリーナモード
            if start_point is None:
                start_point = (x, y)
                cv2.circle(img_copy, start_point, 3, (255, 0, 0), -1)
            elif end_point is None:
                end_point = (x, y)
                draw_circle_arena()
            else:
                start_point = (x, y)
                end_point = None
                cv2.circle(img_copy, start_point, 3, (255, 0, 0), -1)
        else:  # 特定領域モード
            circle_start, circle_end = specific_circles[current_mode - 1] or (None, None)
            if circle_start is None:
                specific_circles[current_mode - 1] = ((x, y), None)
                cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
            elif circle_end is None:
                specific_circles[current_mode - 1] = (circle_start, (x, y))
                draw_specific_circle(current_mode - 1)
            else:
                specific_circles[current_mode - 1] = ((x, y), None)
                cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
        
        # すべての図形を描画
        draw_all_circles()
        cv2.imshow("Image for Arena Selection", img_copy)

    elif event == cv2.EVENT_MOUSEMOVE:
        temp_img = img_copy.copy()
        if current_mode == 0 and start_point is not None and end_point is None:
            center = ((start_point[0] + x) // 2, (start_point[1] + y) // 2)
            radius = int(np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2) // 2)
            cv2.circle(temp_img, center, radius, (255, 0, 0), 2)
        elif current_mode != 0:
            circle_start, circle_end = specific_circles[current_mode - 1] or (None, None)
            if circle_start is not None and circle_end is None:
                center = ((circle_start[0] + x) // 2, (circle_start[1] + y) // 2)
                radius = int(np.sqrt((x - circle_start[0])**2 + (y - circle_start[1])**2) // 2)
                cv2.circle(temp_img, center, radius, (0, 255, 0), 2)
                cv2.circle(temp_img, center, 2, (0, 255, 0), -1)
                cv2.putText(temp_img, str(current_mode), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Image for Arena Selection", temp_img)

def calculate_circle_info(circles, arena_center, arena_radius):
    circle_info = []
    
    # アリーナの情報を追加
    arena_area = np.pi * arena_radius**2
    circle_info.append({
        'name': 'Arena',
        'center_x': arena_center[0],
        'center_y': arena_center[1],
        'radius': arena_radius,
        'area': arena_area,
        'norm_center_x': 0,
        'norm_center_y': 0,
        'norm_radius': 0.5,
        'norm_area': 1
    })
    
    # 特定領域の情報を追加
    for i, circle in enumerate(circles):
        if circle is not None and circle[1] is not None:
            circle_start, circle_end = circle
            specific_center = ((circle_start[0] + circle_end[0]) // 2, (circle_start[1] + circle_end[1]) // 2)
            specific_radius = int(np.sqrt((circle_end[0] - circle_start[0])**2 + (circle_end[1] - circle_start[1])**2) // 2)
            specific_area = np.pi * specific_radius**2
            
            # 正規化された値を計算
            norm_center_x, norm_center_y = normalize_position(specific_center, arena_center, arena_radius)
            norm_radius = specific_radius / (2 * arena_radius)
            norm_area = specific_area / (np.pi * arena_radius**2)
            
            circle_info.append({
                'name': f'Specific_{i+1}',
                'center_x': specific_center[0],
                'center_y': specific_center[1],
                'radius': specific_radius,
                'area': specific_area,
                'norm_center_x': norm_center_x,
                'norm_center_y': norm_center_y,
                'norm_radius': norm_radius,
                'norm_area': norm_area
            })
    
    return circle_info

def draw_circle_arena():
    global img_copy, start_point, end_point
    if start_point is not None and end_point is not None:
        center = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
        radius = int(np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) // 2)
        cv2.circle(img_copy, center, radius, (255, 0, 0), 2)
        cv2.circle(img_copy, center, 2, (255, 0, 0), -1)

def draw_all_circles():
    global img_copy, start_point, end_point, specific_circles
    
    if start_point is not None and end_point is not None:
        draw_circle_arena()
    
    for i, circle in enumerate(specific_circles):
        if circle is not None and circle[1] is not None:
            draw_specific_circle(i)

def draw_specific_circle(index):
    global img_copy, specific_circles
    circle_start, circle_end = specific_circles[index]
    if circle_start is not None and circle_end is not None:
        center = ((circle_start[0] + circle_end[0]) // 2, (circle_start[1] + circle_end[1]) // 2)
        radius = int(np.sqrt((circle_end[0] - circle_start[0])**2 + (circle_end[1] - circle_start[1])**2) // 2)
        cv2.circle(img_copy, center, radius, (0, 255, 0), 2)
        cv2.circle(img_copy, center, 2, (0, 255, 0), -1)
        cv2.putText(img_copy, str(index + 1), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

def draw_arena(image, insect_pos=None):
    if len(image.shape) == 2:
        image_with_arena = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_with_arena = image.copy()
    
    if start_point is not None and end_point is not None:
        cv2.circle(image_with_arena, center, radius, (255, 0, 0), 2)
        cv2.circle(image_with_arena, center, 2, (255, 0, 0), -1)
    
    if insect_pos is not None:
        absolute_pos = (center[0] + insect_pos[0], center[1] - insect_pos[1])
    else:
        absolute_pos = None

    for i, circle in enumerate(specific_circles):
        if circle is not None and circle[1] is not None:
            circle_start, circle_end = circle
            specific_center = ((circle_start[0] + circle_end[0]) // 2, (circle_start[1] + circle_end[1]) // 2)
            specific_radius = int(np.sqrt((circle_end[0] - circle_start[0])**2 + (circle_end[1] - circle_start[1])**2) // 2)
            
            # 昆虫が特定領域内にいるかチェック
            if absolute_pos is not None and is_in_specific_area(absolute_pos, (specific_center, specific_radius)):
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 255, 0)  # 緑色
            
            cv2.circle(image_with_arena, specific_center, specific_radius, color, 2)
            cv2.circle(image_with_arena, specific_center, 2, color, -1)
            cv2.putText(image_with_arena, str(i + 1), specific_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return image_with_arena

def get_centroid_relative_to_specific_circle(insect_pos, specific_circle):
    if insect_pos is None or specific_circle is None:
        return None
    specific_center, _ = specific_circle
    relative_x = insect_pos[0] - specific_center[0]
    relative_y = specific_center[1] - insect_pos[1]  # Y軸を反転
    return np.array([relative_x, relative_y])

##################################################
# ↓ 背景生成
##################################################

# リサイズ関数を追加
def resize_image(image, target_size=(512, 512)):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        new_w = min(target_size[0], w)  # 元のサイズより大きくならないようにする
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = min(target_size[1], h)  # 元のサイズより大きくならないようにする
        new_w = int(new_h * aspect_ratio)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# パーセンタイル画像を生成する関数
def generate_percentile_images(all_frames):
    percentiles = range(10, 91, 5)  # 10, 15, 20, ..., 90
    percentile_frames = []
    for p in percentiles:
        percentile_frame = np.percentile(all_frames, p, axis=0).astype(np.uint8)
        percentile_frames.append(percentile_frame)
    return percentile_frames, percentiles

# GUI クラス
class PercentileSelector:
    def __init__(self, percentile_frames, percentiles):
        self.percentile_frames = percentile_frames
        self.percentiles = percentiles
        self.current_index = 0
        self.selected_frame = None

    def convert_to_bytes(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    def run(self):
        layout = [
            [sg.Image(key="-IMAGE-")],
            [sg.Text("", key="-PERCENTILE-")],
            [sg.Button("Previous"), sg.Button("Next"), sg.Button("Select")]
        ]

        window = sg.Window("Percentile Image Selector", layout, finalize=True)

        self.update_image(window)

        while True:
            event, values = window.read()
            if event == sg.WINDOW_CLOSED:
                break
            elif event == "Previous":
                self.current_index = (self.current_index - 1) % len(self.percentile_frames)
                self.update_image(window)
            elif event == "Next":
                self.current_index = (self.current_index + 1) % len(self.percentile_frames)
                self.update_image(window)
            elif event == "Select":
                self.selected_frame = self.percentile_frames[self.current_index]
                break

        window.close()
        return self.selected_frame

    def update_image(self, window):
        image = self.percentile_frames[self.current_index]
        image = resize_image(image, target_size=(400, 400))
        img_bytes = self.convert_to_bytes(image)
        window["-IMAGE-"].update(data=img_bytes)
        window["-PERCENTILE-"].update(f"Percentile: {self.percentiles[self.current_index]}")



##################################################
# ↓ 閾値調整
##################################################

def adjust_threshold_ratio(sampled_frames, adjusted_background, mask, target_median, target_iqr):
    global threshold_ratio
    def process_frame_for_threshold(frame):
        global threshold_ratio
        adjusted_gray = adjust_brightness(frame, target_median, target_iqr, mask)
        diff = cv2.absdiff(adjusted_gray, adjusted_background)
        denoised = denoise(diff)
        masked_denoised = cv2.bitwise_and(denoised, mask)
        thresh_val = find_threshold(masked_denoised)
        _, masked_thresh = cv2.threshold(masked_denoised, thresh_val, 255, cv2.THRESH_BINARY)
        return masked_denoised, masked_thresh

    def on_trackbar(val):
        global threshold_ratio
        threshold_ratio = val / 10000

    cv2.namedWindow("Threshold Adjustment")
    cv2.createTrackbar("Threshold Ratio", "Threshold Adjustment", 1, 100, on_trackbar)

    frame_index = 0
    num_frames = len(sampled_frames)

    while True:
        frame = sampled_frames[frame_index]
        adjusted_gray, masked_thresh = process_frame_for_threshold(frame)

        # Create a side-by-side display
        display = np.hstack((adjusted_gray, masked_thresh))
        display = resize_image(display, (800,400))
        cv2.putText(display, f"Frame: {frame_index+1}/{num_frames}, Threshold Ratio: {threshold_ratio:.6f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Threshold Adjustment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
        elif key == ord('n'):
            frame_index = (frame_index + 1) % num_frames
        elif key == ord('p'):
            frame_index = (frame_index - 1) % num_frames

    cv2.destroyAllWindows()

##################################################
# ↓ 明度調整関数
##################################################

def calculate_median_brightness(frame, center, radius):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    non_zero_pixels = masked_frame[np.nonzero(masked_frame)]
    if len(non_zero_pixels) > 0:
        return np.median(non_zero_pixels)
    else:
        return 0
    
def adjust_brightness(frame, target_median, target_iqr, mask):
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    non_zero_pixels = masked_frame[np.nonzero(masked_frame)]
    
    if len(non_zero_pixels) > 0:
        current_median = np.median(non_zero_pixels)
        q1 = np.percentile(non_zero_pixels, 25)
        q3 = np.percentile(non_zero_pixels, 75)
        current_iqr = q3 - q1
        
        if current_median > 0 and current_iqr > 0:
            # IQRに基づいてslopeを計算
            slope = target_iqr / current_iqr
            
            # 中央値に基づいてinterceptを計算
            intercept = target_median - (slope * current_median)
            
            # 明度調整を適用
            adjusted_frame = cv2.convertScaleAbs(frame, alpha=slope, beta=intercept)
            
            return adjusted_frame
    
    return frame

def get_iqr(frame, mask=None):
    if mask is not None:
        # マスクのサイズと型を確認し、必要に応じて調整
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # マスクが3チャンネルの場合、1チャンネルに変換
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # マスクが提供された場合、マスクされた領域のみを考慮
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        pixels = masked_frame[np.nonzero(masked_frame)]
    else:
        # マスクがない場合、全ピクセルを使用
        pixels = frame.flatten()
    
    if len(pixels) > 0:
        q1 = np.percentile(pixels, 25)
        q3 = np.percentile(pixels, 75)
        iqr = q3 - q1
        return iqr
    else:
        return 0  # ピクセルがない場合は0を返す

##################################################
# ↑ 明度調整関数
##################################################

##################################################
# ↓ トラッキング
##################################################

def normalize_position(pos, center, radius):
    normalized_x = (pos[0] - center[0]) / (2 * radius)
    normalized_y = (center[1] - pos[1]) / (2 * radius)  # Y軸を反転
    return normalized_x, normalized_y

# 円内にいるかどうかを判定する関数
def is_in_specific_area(point, specific_circle):
    if specific_circle is None:
        return False
    center, radius = specific_circle
    distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    return distance <= radius

def draw_tracked_image(adjusted_gray, insect_contour, insect_pos, insect_trail):
    tracked_image = cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR)
    
    # 昆虫の軌跡を描画
    for i in range(1, len(insect_trail)):
        cv2.line(tracked_image, insect_trail[i - 1], insect_trail[i], (0, 165, 255), 2)

    # 昆虫の楕円を描画
    if insect_contour is not None:
        cv2.drawContours(tracked_image, [insect_contour], 0, (0, 255, 0), 2)

    # 昆虫が検出できていれば、位置に点を描画
    if insect_pos is not None:
        absolute_pos = (center[0] + insect_pos[0], center[1] - insect_pos[1])
        cv2.circle(tracked_image, absolute_pos, 3, (0, 0, 255), -1)

    # アリーナと特定領域を描画
    tracked_image = draw_arena(tracked_image, insect_pos)
    
    return tracked_image

def update_recent_areas(area):
    if len(recent_areas) >= 20:
        recent_areas.pop(0)
    recent_areas.append(area)

def is_outlier(area): # True で除外
    if len(recent_areas) < 20:
        return False  # データが20個未満の場合、除外しない
    mean = np.mean(recent_areas)
    std = np.std(recent_areas)
    return not (mean - 3 * std <= area <= mean + 3 * std)

def denoise(frame):
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def find_threshold(frame):
    global threshold_ratio
    hist, _ = np.histogram(frame.flatten(), 256, [0, 256])
    hist_norm = hist.astype(np.float32) / hist.sum()
    cumulative_sum = np.cumsum(hist_norm)
    threshold_index = np.argmax(cumulative_sum >= 1-threshold_ratio)

    return threshold_index

def find_largest_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def get_centroid_relative_to_arena(contour, arena_center):
    if contour is None:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        relative_x = cx - arena_center[0]
        relative_y = arena_center[1] - cy  # Y軸を反転
        return np.array([relative_x, relative_y])
    return None

def find_insect(binary_image, arena_center):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    
    for contour in contours:
        # 輪郭の面積取得
        area = cv2.contourArea(contour)

        # 輪郭を楕円近似
        if len(contour) >= 5:  # 楕円フィッティングには少なくとも5点が必要
            ellipse = cv2.fitEllipse(contour)
            (center, (major_axis, minor_axis), angle) = ellipse
            
            # 長軸が短軸の5倍未満のもののみを有効とする
            if major_axis / minor_axis >= 5:
                continue

            # 細かいものは除外
            if major_axis < 4 or minor_axis < 3:
                continue

            # 回転楕円の面積を計算
            ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)
            # 面積の比率を計算する
            ratio = area / ellipse_area
            # # 楕円以外は虫ではないと判定
            # if ratio < 0.5:
            #     continue

            # 有効な輪郭として判定
            valid_contours.append(contour)
    
    # 有効な輪郭がなければ、昆虫なしとして戻す
    if not valid_contours:
        return None, None
    
    # 最大の輪郭を見つける
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # 相対座標を計算
    relative_pos = get_centroid_relative_to_arena(largest_contour, arena_center)
    
    return largest_contour, relative_pos

# コマンドライン引数からファイルパスを取得
if len(sys.argv) < 3:
    print("Usage: python nishino_tracking_2.py <input_file> <output_directory>")
    sys.exit(1)

input_file = sys.argv[1]
output_directory = sys.argv[2]

# ファイル名（拡張子なし）を取得
base_name = os.path.splitext(os.path.basename(input_file))[0]

print(input_file)
print(output_directory)
print(base_name)


# 出力ファイル名を生成
output_tracked = os.path.join(output_directory, f"{base_name}_tracked.mp4")
# output_normalized = os.path.join(output_directory, f"{base_name}_normalized.mp4")
# output_binary = os.path.join(output_directory, f"{base_name}_binary.mp4")
# output_diff = os.path.join(output_directory, f"{base_name}_diff.mp4")
output_csv = os.path.join(output_directory, f"{base_name}_tracking_data.csv")
output_circle_info = os.path.join(output_directory, f"{base_name}_circle_info.csv")

# 動画を準備
percentile_frames, percentiles, total_frames = sample_frames_gpu(input_file, num_samples)

def get_percentile_frame(index):
    return cp.asnumpy(cp.percentile(cp.array(percentile_frames), percentiles[index], axis=0).astype(cp.uint8))

# マウスでアリーナを指定
# パーセンタイルの中央値のインデックスを計算
middle_percentile_index = len(percentiles) // 2

# 中央のパーセンタイルフレームを取得
img = get_percentile_frame(middle_percentile_index)

# グレースケールからBGRに変換
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    # 背景画像の視認性を向上
img = enhance_image_visibility(img)
img_copy = img.copy()
cv2.imshow("Image for Arena Selection", img)
cv2.setMouseCallback("Image for Arena Selection", draw_circle)

# 円の外側をマスク
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif ord('0') <= key <= ord('9'):
        current_mode = key - ord('0')
        if current_mode == 0:
            print("アリーナの描画モードに切り替えました。")
        else:
            print(f"特定領域 {current_mode} の描画モードに切り替えました。")
    elif key == 13 and start_point is not None and end_point is not None:
        break

cv2.destroyAllWindows()

# 円形アリーナの定義
center = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
radius = int(np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) // 2)

# 特定領域（緑の円）の定義
if circle_start is not None and circle_end is not None:
    specific_center = ((circle_start[0] + circle_end[0]) // 2, (circle_start[1] + circle_end[1]) // 2)
    specific_radius = int(np.sqrt((circle_end[0] - circle_start[0])**2 + (circle_end[1] - circle_start[1])**2) // 2)
    specific_circle = (specific_center, specific_radius)
else:
    specific_circle = None

# GUIでパーセンタイル画像を選択
selected_percentile_frame = run_selector()

if selected_percentile_frame is None:
    print("No frame was selected. Exiting.")
    exit()

# マスクを作成
mask = np.zeros_like(selected_percentile_frame, dtype=np.uint8)
cv2.circle(mask, center, radius, (255, 255, 255), -1)

target_iqr = get_iqr(selected_percentile_frame, mask)

# 選択された画像を明度調整
adjusted_background = adjust_brightness(selected_percentile_frame, target_median, target_iqr, np.ones_like(selected_percentile_frame, dtype=np.uint8) * 255)

# 閾値調整
adjust_threshold_ratio(sampled_frames, adjusted_background, mask, target_median, target_iqr)

# 動画を準備
cap = cv2.VideoCapture(input_file)

# 出力ビデオの設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_tracked = cv2.VideoWriter(output_tracked, fourcc, fps, (width, height), isColor=True)
# out_normalized = cv2.VideoWriter(output_normalized, fourcc, fps, (width, height), isColor=True)
# out_binary = cv2.VideoWriter(output_binary, fourcc, fps, (width, height), isColor=True)
# out_diff = cv2.VideoWriter(output_diff, fourcc, fps, (width, height), isColor=True)

if not out_tracked.isOpened():
    print("Error: Could not open one or more output video files.")
    exit()
# if not out_normalized.isOpened() or not out_binary.isOpened() or not out_diff.isOpened():
#     print("Error: Could not open one or more output video files.")
#     exit()

try:
    # プログレスバーの設定
    pbar = tqdm(total=total_frames, desc=f"Processing {base_name}", unit="frame")

    frame_queue = queue.Queue(maxsize=30)  # フレームを保持するキュー
    result_queue = queue.Queue()  # 処理結果を保持するキュー

    def frame_producer():
        try:
            with tqdm(total=total_frames, desc="Reading frames", unit="frame") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        frame_queue.put(None)  # 終了信号
                        break
                    frame_queue.put(frame)
                    pbar.update(1)
        except Exception as e:
            print(f"Producer error: {str(e)}")
            frame_queue.put(None)  # エラー時も終了信号を送る

    def frame_consumer():
        try:
            with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
                while True:
                    frame = frame_queue.get()
                    if frame is None:
                        result_queue.put(None)  # 終了信号
                        break
                    result = process_frame(frame, adjusted_background, mask, target_median, target_iqr)
                    result_queue.put(result)
                    pbar.update(1)
        except Exception as e:
            print(f"Consumer error: {str(e)}")
            result_queue.put(None)  # エラー時も終了信号を送る

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        producer_future = executor.submit(frame_producer)
        consumer_future = executor.submit(frame_consumer)

        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        with tqdm(total=total_frames, desc="Writing output", unit="frame") as pbar:
            while True:
                try:
                    result = result_queue.get(timeout=10)  # タイムアウトを設定
                    if result is None:
                        break

                    adjusted_gray, diff, masked_thresh = result

                    # 昆虫の検出とトラッキング（順序を保つために逐次的に実行）
                    insect_contour, insect_pos = find_insect(masked_thresh, center)

                    # insect_trailの更新
                    if insect_pos is not None:
                        absolute_pos = (center[0] + insect_pos[0], center[1] - insect_pos[1])
                        insect_trail.append(absolute_pos)

                    # 結果の描画と出力（順序を保つために逐次的に実行）
                    tracked_image = draw_tracked_image(adjusted_gray, insect_contour, insect_pos, insect_trail)
                    normalized_image = draw_arena(cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR), insect_pos)
                    diff_image = draw_arena(cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), insect_pos)
                    binary_image = draw_arena(cv2.cvtColor(masked_thresh, cv2.COLOR_GRAY2BGR), insect_pos)

                    # 書き込み
                    out_tracked.write(tracked_image)
                    # out_normalized.write(normalized_image)
                    # out_diff.write(diff_image)
                    # out_binary.write(binary_image)

                    # データの記録
                    current_time = frame_count / fps  # 動画内での経過時間（秒）
                    if insect_pos is not None:
                        # 絶対座標
                        absolute_pos = (center[0] + insect_pos[0], center[1] - insect_pos[1])
                        # アリーナ中心からの相対座標
                        arena_relative_x, arena_relative_y = insect_pos
                        # アリーナ中心を(0,0)とし、直径を1とした正規化位置
                        norm_arena_x, norm_arena_y = normalize_position(absolute_pos, center, radius)

                        # 特定領域の中心からの相対座標
                        specific_data = []
                        for i, circle in enumerate(specific_circles):
                            if circle is not None and circle[1] is not None:
                                specific_center = ((circle[0][0] + circle[1][0]) // 2, (circle[0][1] + circle[1][1]) // 2)
                                specific_radius = int(np.sqrt((circle[1][0] - circle[0][0])**2 + (circle[1][1] - circle[0][1])**2) // 2)
                                specific_relative_pos = get_centroid_relative_to_specific_circle(absolute_pos, (specific_center, specific_radius))
                                in_specific_area = is_in_specific_area(absolute_pos, (specific_center, specific_radius))
                                
                                # 特定領域の中心を(0,0)とし、直径を1とした正規化位置
                                norm_specific_x, norm_specific_y = normalize_position(absolute_pos, specific_center, specific_radius)
                                
                                specific_data.extend([
                                    specific_relative_pos[0], specific_relative_pos[1],
                                    norm_specific_x, norm_specific_y,
                                    int(in_specific_area)
                                ])
                            else:
                                specific_data.extend(["None", "None", "None", "None", "None"])
                        
                        insect_data.append([current_time, arena_relative_x, arena_relative_y, norm_arena_x, norm_arena_y] + specific_data)
                    else:
                        insect_data.append([current_time, None, None, None, None] + [None] * (5 * 9))

                    frame_count += 1
                    pbar.update(1)
                except queue.Empty:
                    print("Timeout waiting for results. Breaking loop.")
                    break
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    traceback.print_exc()  # スタックトレースを出力
                    continue  # エラーが発生しても次のフレームに進む

        # futures の結果を取得し、エラーがあれば報告
        try:
            producer_future.result()
        except Exception as e:
            print(f"Producer future error: {str(e)}")

        try:
            consumer_future.result()
        except Exception as e:
            print(f"Consumer future error: {str(e)}")

except Exception as e:
    print(f"Main loop error: {str(e)}")
    traceback.print_exc()  # スタックトレースを出力

finally:
    # リソースの解放
    cap.release()
    out_tracked.release()
    # out_normalized.release()
    # out_binary.release()
    # out_diff.release()
    cv2.destroyAllWindows()

    # 時系列データをCSVファイルに保存
    columns = ['time', 'arena_relative_x', 'arena_relative_y', 'norm_arena_x', 'norm_arena_y']
    for i in range(9):
        columns.extend([
            f'specific_{i+1}_relative_x',
            f'specific_{i+1}_relative_y',
            f'specific_{i+1}_norm_x',
            f'specific_{i+1}_norm_y',
            f'in_specific_area_{i+1}'
        ])

    df = pd.DataFrame(insect_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

    # Circle情報の計算と保存
    circle_info = calculate_circle_info(specific_circles, center, radius)
    circle_info_df = pd.DataFrame(circle_info)
    circle_info_df.to_csv(output_circle_info, index=False)
    print(f"Circle information saved to {output_circle_info}")
