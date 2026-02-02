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
        Video tensor of shape [T, C, H, W] normalized to [-1, 1] on specified device,
        or None if video cannot be loaded
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads in BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if target_size is not None:
                # cv2.resize expects (width, height), while target_size is usually (height, width)
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
            
            frames.append(frame)
    finally:
        cap.release()
        
    if not frames:
        return None
    
    video_np = np.stack(frames)  # Shape: [T, H, W, C]
    
    # Convert to tensor and move to device
    tensor = torch.from_numpy(video_np).to(device, non_blocking=True)
    
    # [T, H, W, C] -> [T, C, H, W]
    tensor = tensor.permute(0, 3, 1, 2).float()
    
    # Normalize to [-1, 1]
    # Original [0, 255] -> [0, 1]: x / 255.0
    # New [0, 255] -> [-1, 1]: (x / 127.5) - 1.0
    tensor = (tensor / 127.5) - 1.0
    
    return tensor