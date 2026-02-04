import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
import random
import logging
from dataclasses import dataclass
from typing import Dict, Any
from decord import VideoReader, cpu

from .env import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class Options:
    """Runtime configuration options."""
    repo_id:        str  = CONFIG.dataset.repo_id
    root:           str  = CONFIG.dataset.root
    index:          str  = CONFIG.dataset.index
    phase:          str  = CONFIG.dataset.phase
    clip_len:       int  = CONFIG.dataset.clip_len
    stride:         int  = CONFIG.dataset.stride
    height:         int  = CONFIG.dataset.height   
    width:          int  = CONFIG.dataset.width
    batch_size:     int  = CONFIG.dataset.batch_size                                      
    serial_batches: bool = CONFIG.dataset.serial_batches                             
    num_workers:    int  = CONFIG.dataset.num_workers

class FollowBenchDatasetWrapper(Dataset):
    def __init__(self, opt: Options) -> None:
        self.opt = opt
        self.data_root = os.path.join(self.opt.root, self.opt.phase)
        self.dataset = self._load_index()
        self.ids = list(self.dataset.keys())
        
        # 图像/视频的后处理变换 (读取后执行)
        self.pixel_transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _load_index(self) -> dict:
        index_path = os.path.join(self.data_root, self.opt.index)
        logger.info(f"Loading index from {index_path}...")
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Index file not found at {index_path}")
            return {}

    def _get_video_reader(self, rel_path):
        """安全获取 VideoReader"""
        path = os.path.join(self.data_root, rel_path)
        try:
            # num_threads=1 避免多线程死锁，ctx=cpu(0) 使用 CPU 解码
            vr = VideoReader(path, ctx=cpu(0), num_threads=1)
            return vr, path
        except Exception as e:
            # logger.warning(f"Failed to load video {path}: {e}")
            return None, path

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        参考 WorldWander 实现：
        1. 容错重试循环
        2. 共享随机索引 (Shared Indexing)
        3. Decord 批量读取
        """
        # 确保索引在范围内
        index = index % len(self.ids)
        
        retries = 0
        while True:
            # 防止死循环
            if retries > 10:
                raise RuntimeError("Too many bad videos in dataset! Check your data.")
            
            vid_id = self.ids[index]
            data = self.dataset[vid_id]
            
            # 1. 加载两个视频的 Reader (此时不读取数据，只读元数据)
            ego_reader, ego_path = self._get_video_reader(data['the first view'])
            exo_reader, exo_path = self._get_video_reader(data['the third view'])
            
            # 2. 验证视频有效性
            if ego_reader is None or exo_reader is None:
                # 视频损坏，随机换一个索引重试
                index = random.randint(0, len(self.ids) - 1)
                retries += 1
                continue
                
            ego_len = len(ego_reader)
            exo_len = len(exo_reader)
            
            # 3. 验证长度一致性 (WorldWander 的核心逻辑)
            # 要求两个视频长度差异不能太大，且必须长于需要的采样长度
            min_len = min(ego_len, exo_len)
            needed_len = (self.opt.clip_len - 1) * self.opt.stride + 1
            
            # 如果视频太短，或者两者长度严重不一致(比如差了100帧以上，说明物理没对齐)
            # 这里宽松一点：只要都能切出需要的长度即可
            if min_len < needed_len:
                # logger.warning(f"Video too short: {vid_id} (len={min_len}, need={needed_len})")
                index = random.randint(0, len(self.ids) - 1)
                retries += 1
                continue
            
            # 4. 计算共享切片索引 (Core Synchronization Logic)
            try:
                # 随机选择起始点
                max_start_idx = min_len - needed_len
                if self.opt.phase == 'train':
                    start_idx = random.randint(0, max_start_idx)
                else:
                    start_idx = 0 # 测试时固定从头开始，或者取中间
                
                # 生成帧索引序列: [start, start+stride, start+2*stride, ...]
                batch_indices = start_idx + np.arange(self.opt.clip_len) * self.opt.stride
                
                # 5. 使用 Decord 批量读取 (Get Batch)
                # get_batch 返回的是 (T, H, W, C) 的 tensor/array
                ego_frames = ego_reader.get_batch(batch_indices).asnumpy()
                exo_frames = exo_reader.get_batch(batch_indices).asnumpy()
                
                # 6. 转换为 PyTorch Tensor 并处理
                # WorldWander 是在这里做 Resize 和 Normalize
                ego_tensor = self._process_frames(ego_frames) # [C, T, H, W]
                exo_tensor = self._process_frames(exo_frames) # [C, T, H, W]
                
                # 成功获取！
                break
                
            except Exception as e:
                logger.warning(f"Error processing video {vid_id}: {e}")
                index = random.randint(0, len(self.ids) - 1)
                retries += 1
                continue

        # 7. 加载参考图 (如果有)
        ref_path = data.get('reference', None)
        if ref_path:
            ref_tensor = self._load_image(ref_path)
        else:
            ref_tensor = torch.zeros(3, self.opt.height, self.opt.width)

        return {
            'video_id': vid_id,
            'source_video': ego_tensor,
            'target_video': exo_tensor,
            'prompt': CONFIG.dataset.instruction,
            'ref_image': ref_tensor,
        }

    def _process_frames(self, frames_np):
        """
        Args:
            frames_np: numpy array (T, H, W, C), usually RGB uint8
        Returns:
            tensor: (C, T, H, W), normalized [-1, 1]
        """
        # 转为 list of PIL Image 以利用 torchvision transforms
        # 也可以直接用 tensor 操作，但保持和 _load_image 一致
        tensors = []
        for i in range(frames_np.shape[0]):
            img = Image.fromarray(frames_np[i])
            tensors.append(self.pixel_transform(img))
        
        # Stack -> (T, C, H, W)
        video_tensor = torch.stack(tensors)
        
        # Permute -> (C, T, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor

    def _load_image(self, rel_path: str):
        path = os.path.join(self.data_root, rel_path)
        try:
            img = Image.open(path).convert('RGB')
            return self.pixel_transform(img)
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return torch.zeros(3, self.opt.height, self.opt.width)
        
class Style1000DatasetWrapper(Dataset):
    """
    专为 Style1000 设计的通用数据集加载器。
    """
    def __init__(self, opt: Options):
        self.opt = opt
        # data_root 指向 style1000 文件夹
        self.data_root = opt.root 
        self.dataset = self._load_index(opt.index)
        
        # 预处理管线：与 Wan2.1 VAE 对齐
        self.pixel_transform = transforms.Compose([
            transforms.Resize((opt.height, opt.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _load_index(self, index_path):
        # 支持绝对路径或相对于 root 的路径
        if not os.path.isabs(index_path):
            index_path = os.path.join(self.data_root, index_path)
            
        logger.info(f"📂 Loading style1000 index from: {index_path}")
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 确保是列表格式
            return data if isinstance(data, list) else []

    def __len__(self):
        return len(self.dataset)

    def _get_video_tensor(self, rel_path, start_idx):
        path = os.path.join(self.data_root, rel_path)
        vr = VideoReader(path, ctx=cpu(0))
        
        # 计算采样帧索引序列
        batch_indices = start_idx + np.arange(self.opt.clip_len) * self.opt.stride
        frames_np = vr.get_batch(batch_indices).asnumpy()
        
        # 处理每一帧
        tensors = []
        for i in range(frames_np.shape[0]):
            img = Image.fromarray(frames_np[i])
            tensors.append(self.pixel_transform(img))
            
        # 堆叠并转换为 [C, T, H, W]
        return torch.stack(tensors).permute(1, 0, 2, 3)

    def __getitem__(self, index):
        retries = 0
        while retries < 10:
            try:
                item = self.dataset[index]
                
                # 映射字段：destyle (x0) -> style (x1)
                source_path = item['destyle']
                target_path = item['style']
                prompt = item.get('caption', "")

                # 获取视频读取器以确定长度
                vr_source = VideoReader(os.path.join(self.data_root, source_path))
                vr_target = VideoReader(os.path.join(self.data_root, target_path))
                
                min_len = min(len(vr_source), len(vr_target))
                needed = (self.opt.clip_len - 1) * self.opt.stride + 1
                
                if min_len < needed:
                    raise ValueError("Video too short")

                # [核心逻辑] 共享随机起点，确保物理时间对齐
                max_start = min_len - needed
                start_idx = random.randint(0, max_start) if self.opt.phase == 'train' else 0
                
                source_video = self._get_video_tensor(source_path, start_idx)
                target_video = self._get_video_tensor(target_path, start_idx)

                return {
                    "source_video": source_video,   # x0
                    "target_video": target_video,   # x1
                    "prompt": prompt,
                    "video_id": index
                }
            except Exception as e:
                index = random.randint(0, len(self.dataset) - 1)
                retries += 1
                continue
        
        raise RuntimeError("Failed to load data after multiple retries.")