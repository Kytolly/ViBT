import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import torch
from torch import Tensor

def load_video_to_device(
    video_path: Union[str, Path], 
    target_size: Optional[Tuple[int, int]] = None, 
    max_frames: Optional[int] = None, 
    device: str = 'cuda',
    sample_stride: int = 4  # [新增] 采样步长：默认为4 (即 60fps -> 15fps)
) -> Optional[Tensor]:
    """
    sample_stride: 每隔多少帧取一帧。
                   如果原视频 60fps, stride=4, 则输入给模型的是 15fps。
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # [新增] 抽帧逻辑：只保留符合 stride 的帧
            if frame_idx % sample_stride == 0:
                # OpenCV reads in BGR, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if target_size is not None:
                    frame = cv2.resize(frame, (target_size[1], target_size[0]))
                frames.append(frame)
            
            frame_idx += 1
            
    finally:
        cap.release()
        
    if not frames:
        return None
    
    # 截断或填充逻辑 (保持不变)
    if max_frames is not None:
        total_frames = len(frames)
        if total_frames > max_frames:
            frames = frames[:max_frames]
        elif total_frames < max_frames:
            while len(frames) < max_frames:
                frames.extend(frames[:max_frames - len(frames)])
    
    video_np = np.stack(frames)
    tensor = torch.from_numpy(video_np).to(device, non_blocking=True)
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = (tensor / 127.5) - 1.0
    
    return tensor