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
import logging
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
        self.resize_transform = transforms.Resize((opt.height, opt.width))

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
        # 使用 CPU 解码
        vr = VideoReader(path, ctx=cpu(0))
        
        batch_indices = start_idx + np.arange(self.opt.clip_len) * self.opt.stride
        frames_np = vr.get_batch(batch_indices).asnumpy() # [T, H, W, C] (uint8)
        
        # [优化] 手动处理：Numpy -> PIL Resize -> Tensor (Uint8)
        tensors = []
        for i in range(frames_np.shape[0]):
            img = Image.fromarray(frames_np[i])
            img_resized = self.resize_transform(img)
            # PIL -> Torch Tensor (uint8, 0-255, [C, H, W])
            # 注意：transforms.ToTensor() 会自动除以 255 转 float，所以我们用 torch.from_numpy
            # 但为了兼容 resize 后的 PIL 对象，最快的方式是：
            # PIL Resize -> np.array -> torch.from_numpy
            
            # 这种方式最稳健：
            tensors.append(torch.from_numpy(np.array(img_resized)))
            
        # Stack -> [T, H, W, C]
        video_tensor = torch.stack(tensors)
        
        # Permute -> [C, T, H, W]
        video_tensor = video_tensor.permute(3, 0, 1, 2).contiguous()
        
        # 此时 video_tensor 是 uint8 类型，范围 [0, 255]
        return video_tensor

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
    
class InMemoryLatentDataset(Dataset):
    def __init__(self, root_dir, cache_folder="latents_cache_v3_norm"):
        self.cache_dir = os.path.join(root_dir, cache_folder)
        self.data = []
        
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache dir not found: {self.cache_dir}. Please run prepare_latents.py first.")
            
        files = sorted([f for f in os.listdir(self.cache_dir) if f.endswith('.pt')])
        logger.info(f"🚀 Loading {len(files)} latents into RAM...")
        
        # 即使 2000 个视频，Latent 总大小也就 3GB 左右，完全可以常驻内存
        for f in tqdm(files, desc="Loading Cache"):
            try:
                # map_location='cpu' 确保不占用显存
                self.data.append(torch.load(os.path.join(self.cache_dir, f), map_location='cpu'))
            except Exception as e:
                logger.warning(f"Error loading {f}: {e}")
                
        logger.info(f"✅ Loaded {len(self.data)} samples ready for training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 零 IO 开销
        return self.data[index]

# ==========================================
# 核心：自动化预处理逻辑
# ==========================================
def ensure_latents_cached(cfg, opt):
    """
    检查 Latent 缓存，如果不存在则自动生成。
    """
    # 定义缓存路径 (包含关键参数以防混淆)
    cache_name = f"latents_cache_{opt.clip_len}f_{opt.height}x{opt.width}_v2"
    cache_dir = os.path.join(opt.root, cache_name)
    
    # 1. 检查是否已完成 (简单检查文件数量是否足够)
    # 这里假设 Style1000 约有 1000+ 数据
    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 100:
        logger.info(f"✅ Cache hit: {cache_dir}. Skipping generation.")
        return cache_dir

    # 2. 如果未完成，开始生成
    logger.info(f"⚠️ Cache missing at {cache_dir}. Starting JIT preprocessing...")
    os.makedirs(cache_dir, exist_ok=True)

    # 临时加载 DatasetWrapper (读取原始视频)
    # 注意：在这里我们可以使用最优化的配置来跑预处理
    from .dataset_wrapper import Style1000DatasetWrapper # 引用自身
    raw_dataset = Style1000DatasetWrapper(opt)
    raw_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False, num_workers=8)

    # 加载 VAE (仅 VAE)
    logger.info("loading VAE for preprocessing...")
    device = "cuda"
    model = WanModel.from_pretrained(cfg.model.path, device=device, dtype=torch.bfloat16)
    model.transformer = None # 卸载 Transformer 节省显存
    
    logger.info(f"🔢 Processing {len(raw_dataset)} videos...")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(raw_loader), total=len(raw_loader)):
            # 拿到原始数据 (uint8)
            # 归一化 [-1, 1]
            src = batch['source_video'].to(device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            tgt = batch['target_video'].to(device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            
            # Encode + Norm
            def _enc(x):
                dist = model.vae.encode(x).latent_dist
                z = dist.mode()
                if hasattr(model.vae.config, "latents_mean"):
                    mean = torch.tensor(model.vae.config.latents_mean).to(z)
                    std = torch.tensor(model.vae.config.latents_std).to(z)
                    z = (z - mean) / std
                return z.cpu()

            z_src = _enc(src)
            z_tgt = _enc(tgt)
            
            # 保存
            video_id = batch['video_id'][0].item()
            torch.save({
                "source": z_src.squeeze(0),
                "target": z_tgt.squeeze(0),
                "prompt": batch['prompt'][0],
                "id": video_id
            }, os.path.join(cache_dir, f"{video_id}.pt"))
            
    # 清理 VAE 显存，为后续训练腾地
    del model
    torch.cuda.empty_cache()
    logger.info("✅ Preprocessing done. Starting training...")
    
    return cache_dir