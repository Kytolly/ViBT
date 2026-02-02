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
    sample_stride: int = 4 
) -> Optional[Tensor]:
    """
    Returns:
        Video tensor of shape [C, T, H, W] normalized to [-1, 1].
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if target_size is not None:
                    # cv2.resize uses (width, height)
                    frame = cv2.resize(frame, (target_size[1], target_size[0]))
                frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()
        
    if not frames:
        return None
    
    if max_frames is not None:
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        elif len(frames) < max_frames:
            while len(frames) < max_frames:
                frames.extend(frames[:max_frames - len(frames)])
    
    video_np = np.stack(frames) # [T, H, W, C]
    
    tensor = torch.from_numpy(video_np).to(device, non_blocking=True)
    tensor = tensor.permute(3, 0, 1, 2).float()
    
    # Normalize to [-1, 1]
    tensor = (tensor / 127.5) - 1.0
    
    return tensor