import torch
from torch.utils.data import Dataset, DataLoader
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
from tqdm import tqdm

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

# ==========================================
# 1. 基础 Dataset (仅用于预处理读取)
# ==========================================
class Style1000DatasetWrapper(Dataset):
    def __init__(self, opt: Options):
        self.opt = opt
        self.data_root = opt.root 
        self.dataset = self._load_index(opt.index)
        # 预处理管线: 仅 Resize，保持 uint8，Norm 留给 GPU
        self.resize_transform = transforms.Resize((opt.height, opt.width))

    def _load_index(self, index_path):
        if not os.path.isabs(index_path):
            index_path = os.path.join(self.data_root, index_path)
        logger.info(f"📂 Loading style1000 index from: {index_path}")
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def __len__(self):
        return len(self.dataset)

    def _get_video_tensor(self, rel_path, start_idx):
        path = os.path.join(self.data_root, rel_path)
        vr = VideoReader(path, ctx=cpu(0))
        batch_indices = start_idx + np.arange(self.opt.clip_len) * self.opt.stride
        frames_np = vr.get_batch(batch_indices).asnumpy()
        
        tensors = []
        for i in range(frames_np.shape[0]):
            img = Image.fromarray(frames_np[i])
            img_resized = self.resize_transform(img)
            tensors.append(torch.from_numpy(np.array(img_resized)))
        
        # [C, T, H, W] uint8
        return torch.stack(tensors).permute(3, 0, 1, 2).contiguous()

    def __getitem__(self, index):
        retries = 0
        while retries < 10:
            try:
                item = self.dataset[index]
                source_path = item['destyle']
                target_path = item['style']
                prompt = item.get('caption', "")

                vr_source = VideoReader(os.path.join(self.data_root, source_path))
                min_len = len(vr_source)
                needed = (self.opt.clip_len - 1) * self.opt.stride + 1
                
                if min_len < needed: raise ValueError("Too short")

                max_start = min_len - needed
                start_idx = random.randint(0, max_start) if self.opt.phase == 'train' else 0
                
                return {
                    "source_video": self._get_video_tensor(source_path, start_idx),
                    "target_video": self._get_video_tensor(target_path, start_idx),
                    "prompt": prompt,
                    "video_id": index
                }
            except Exception:
                index = random.randint(0, len(self.dataset) - 1)
                retries += 1
        raise RuntimeError("Failed load")

# ==========================================
# 2. 内存 Latent Dataset (用于训练)
# ==========================================
class InMemoryLatentDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.data = []
        
        # 扫描所有 .pt 文件
        files = sorted([f for f in os.listdir(self.cache_dir) if f.endswith('.pt')])
        logger.info(f"🚀 Loading {len(files)} latents into RAM from {self.cache_dir}...")
        
        # 全量加载到 CPU 内存
        for f in tqdm(files, desc="Loading Cache"):
            try:
                self.data.append(torch.load(os.path.join(self.cache_dir, f), map_location='cpu'))
            except Exception as e:
                logger.warning(f"Bad file {f}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# ==========================================
# 3. JIT 自动化预处理函数
# ==========================================
def ensure_latents_cached(cfg, opt):
    """
    检查 Latent 缓存，如果不存在则自动生成。
    """
    cache_name = f"latents_cache_{opt.clip_len}f_{opt.height}x{opt.width}_v2"
    cache_dir = os.path.join(opt.root, cache_name)
    
    # 1. 检查是否存在
    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 100:
        logger.info(f"✅ Cache hit: {cache_dir}. Skipping generation.")
        return cache_dir

    # 2. 不存在则生成
    logger.info(f"⚠️ Cache missing at {cache_dir}. Starting JIT preprocessing...")
    os.makedirs(cache_dir, exist_ok=True)

    # 临时加载 Raw Dataset
    raw_dataset = Style1000DatasetWrapper(opt)
    raw_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False, num_workers=8)

    logger.info("loading VAE for preprocessing...")
    from .wan import WanModel # 延迟导入避免循环引用
    device = "cuda"
    # 仅加载 VAE
    model = WanModel.from_pretrained(cfg.model.path, device=device, dtype=torch.bfloat16)
    model.transformer = None 
    
    logger.info(f"🔢 Processing {len(raw_dataset)} videos...")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(raw_loader), total=len(raw_loader)):
            # Pixel Norm: uint8 -> [-1, 1]
            src = batch['source_video'].to(device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            tgt = batch['target_video'].to(device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            
            # Encode + Latent Norm
            def _enc(x):
                dist = model.vae.encode(x).latent_dist
                z = dist.mode()
                if hasattr(model.vae.config, "latents_mean"):
                    mean = torch.tensor(model.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
                    std = torch.tensor(model.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
                    z = (z - mean) / std
                return z.cpu()

            z_src = _enc(src)
            z_tgt = _enc(tgt)
            
            video_id = batch['video_id'][0].item()
            torch.save({
                "source": z_src.squeeze(0),
                "target": z_tgt.squeeze(0),
                "prompt": batch['prompt'][0],
                "id": video_id
            }, os.path.join(cache_dir, f"{video_id}.pt"))
            
    del model
    torch.cuda.empty_cache()
    logger.info("✅ Preprocessing done. Starting training...")
    
    return cache_dir