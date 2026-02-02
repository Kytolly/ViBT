import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path

import torch
from torch import Tensor

def load_video_to_device(
    video_path: Union[str, Path], 
    target_size: Optional[Tuple[int, int]] = None, 
    device: str = 'cuda'
) -> Optional[Tensor]:
    """Load video directly to specified device as tensor.
    
    Reads video frames, optionally resizes them, and converts to tensor
    format [T, C, H, W] on the specified device.
    
    Args:
        video_path: Path to the video file
        target_size: Optional target size as (height, width) for resizing
        device: Target device ('cuda' or 'cpu')
        
    Returns:
        Video tensor of shape [T, C, H, W] normalized to [0, 1] on specified device,
        or None if video cannot be loaded
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
            frames.append(frame)
    finally:
        cap.release()
        
    if not frames:
        return None
    
    video_np = np.stack(frames)  # [T, H, W, C]
    tensor = torch.from_numpy(video_np).to(device, non_blocking=True)
    
    # [T, H, W, C] -> [T, C, H, W] & Normalize to [0, 1]
    tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
    return tensor