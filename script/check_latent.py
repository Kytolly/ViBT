import cv2
import numpy as np

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    # 你的视频是 [Source | Pred | GT] 三屏拼接
    full_h, full_w, _ = frames[0].shape
    w = full_w // 3

    first_pred = frames[0][:, w:2*w, :]
    last_pred = frames[-1][:, w:2*w, :]
    source = frames[0][:, :w, :]

    # 计算偏移量
    diff_from_source = np.abs(last_pred.astype(float) - source.astype(float)).mean()
    noise_level = np.std(last_pred)
    
    print(f"Pred末帧相对于Source的偏移量: {diff_from_source:.2f}")
    print(f"Pred末帧噪声标准差: {noise_level:.2f} (若显著高于Source，说明模型在发散)")

analyze_video("outputs/stylization/samples/val_step_16000_combined.mp4")