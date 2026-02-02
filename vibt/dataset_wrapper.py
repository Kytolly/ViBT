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
    """
    Runtime configuration options.
    """
    # 基础信息
    repo_id:        str  = CONFIG.dataset.repo_id # HuggingFace 仓库 ID
    
    # 路径配置
    root:           str  = CONFIG.dataset.root                              # 数据集本地根目录
    index:          str  = CONFIG.dataset.index
    
    # 运行模式
    phase:          str  = CONFIG.dataset.phase                                 # train | test_seen | test_unseen

    clip_len:       int  = CONFIG.dataset.clip_len            
    height:         int  = CONFIG.dataset.height   
    width:          int  = CONFIG.dataset.width
    
    # Dataloader 配置
    batch_size:     int  = CONFIG.dataset.batch_size                                      
    serial_batches: bool = CONFIG.dataset.serial_batches                             
    num_workers:    int  = CONFIG.dataset.num_workers

class FollowBenchDatasetWrapper(Dataset):
    def __init__(self, opt: Options) -> None:
        """Initialize dataset.
        Args:
            opt: Options object with runtime configuration.
        """
        self.opt = opt
        self.annotation: Dict[str, Any] = {}
        self.transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_root = os.path.join(self.opt.root, self.opt.phase)
        
        # 2. 加载索引
        self.dataset = self._load_index()
        self.ids = list(self.dataset.keys())

    def _load_index(self) -> None:
        """Load index for the selected phase."""
        index_path = os.path.join(self.data_root, self.opt.index)
        logger.info(f"Loading index from {index_path}...")
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                return dataset
        except FileNotFoundError:
            logger.warning(f"Index file not found at {index_path}")
            return {}
        
    def _load_image(self, rel_path: str) -> Tensor:
        """Load an image and apply transforms.

        Args:
            rel_path: Relative path to the image inside data root.

        Returns:
            Transformed image tensor of shape [C, H, W].
        """
        path = os.path.join(self.data_root, rel_path)
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img) # [Fix] Apply transform
        except Exception as e:
            logging.error(f"Failed to load image {path}: {e}")
            return torch.zeros(3, self.opt.height, self.opt.width)
    
    def _load_video(self, rel_path: str) -> "torch.Tensor":
        """Load a video and return a tensor.

        Args:
            rel_path: Relative path to the video inside data root.

        Returns:
            Video tensor of shape [T, C, H, W].
        """
        path = os.path.join(self.data_root, rel_path)
        try:
            return load_video_to_device(path, device='cpu') 
        except Exception as e:
            logging.error(f"Failed to load video {path}: {e}")
            return torch.zeros(self.opt.clip_len, 3, self.opt.height, self.opt.width)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a single sample by index.

        Args:
            index: Index of the sample.

        Returns:
            A dict containing 'video_id', 'ego_video', 'exo_video', and 'ref_image'.
        """
        vid_id = self.ids[index]
        data = self.dataset[vid_id]
        
        ego_video = self._load_video(data['the first view'])
        exo_video = self._load_video(data['the third view']) # GT
        ref_img = self._load_image(data['reference'])
        
        return {
            'video_id': vid_id,
            'ego_video': ego_video,
            'exo_video': exo_video, # GT
            'ref_image': ref_img,
        }