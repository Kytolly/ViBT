import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch import Tensor

import os
import json
from dataclasses import dataclass
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

from .utils import load_video_to_device
from .env import CONFIG

@dataclass
class Options:
    """Runtime configuration options."""
    repo_id:        str  = CONFIG.dataset.repo_id
    root:           str  = CONFIG.dataset.root
    index:          str  = CONFIG.dataset.index
    phase:          str  = CONFIG.dataset.phase
    clip_len:       int  = CONFIG.dataset.clip_len            
    height:         int  = CONFIG.dataset.height   
    width:          int  = CONFIG.dataset.width
    batch_size:     int  = CONFIG.dataset.batch_size                                      
    serial_batches: bool = CONFIG.dataset.serial_batches                             
    num_workers:    int  = CONFIG.dataset.num_workers
    instruction:    str  = getattr(CONFIG.dataset, "instruction", "Transform the video")

class FollowBenchDatasetWrapper(Dataset):
    def __init__(self, opt: Options) -> None:
        self.opt = opt
        self.transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_root = os.path.join(self.opt.root, self.opt.phase)
        self.dataset = self._load_index()
        self.ids = list(self.dataset.keys())

    def _load_index(self) -> dict:
        index_path = os.path.join(self.data_root, self.opt.index)
        print(f"Loading index from {index_path}...")
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Index file not found at {index_path}")
            return {}
        
    def _load_image(self, rel_path: str) -> Tensor:
        path = os.path.join(self.data_root, rel_path)
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            logging.error(f"Failed to load image {path}: {e}")
            return torch.zeros(3, self.opt.height, self.opt.width)
    
    def _load_video(self, rel_path: str) -> Tensor:
        path = os.path.join(self.data_root, rel_path)
        try:
            # [核心修复 1] 传递 target_size，确保 resize 生效
            return load_video_to_device(
                path, 
                device='cpu', 
                max_frames=self.opt.clip_len,
                target_size=(self.opt.height, self.opt.width) 
            ) 
        except Exception as e:
            logging.error(f"Failed to load video {path}: {e}")
            # [核心修复 2] 失败时返回正确维度的 Tensor [C, T, H, W]
            return torch.zeros(3, self.opt.clip_len, self.opt.height, self.opt.width)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        vid_id = self.ids[index]
        data = self.dataset[vid_id]
        
        return {
            'video_id': vid_id,
            'ego_video': self._load_video(data['the first view']),
            'exo_video': self._load_video(data['the third view']),
            'ref_image': self._load_image(data['reference']),
        }