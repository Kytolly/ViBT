import os
import sys
import torch
import torchvision
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging

# 添加项目根目录到 python path
sys.path.append(os.getcwd())

from vibt.dataset_wrapper import Style1000DatasetWrapper, Options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_video_smart(tensor, save_path, fps=8):
    """
    智能保存视频，自动处理 uint8 和 float 输入。
    输入 Tensor 形状: [C, T, H, W]
    """
    video = tensor.clone().detach()
    
    # 1. 检查数据类型并处理
    if video.dtype == torch.uint8:
        # 情况 A: 已经是 [0, 255] 的 uint8 (Dataset 优化后的输出)
        logger.info(f"   Detected uint8 input [0, 255]. Saving directly.")
        pass # 无需处理
        
    elif video.is_floating_point():
        # 情况 B: [-1, 1] 的 float (旧 Dataset 或模型输出)
        logger.info(f"   Detected float input (min={video.min():.2f}, max={video.max():.2f}). Unnormalizing...")
        video = video * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        video = torch.clamp(video, 0, 1) * 255
        video = video.to(torch.uint8)
        
    # 2. 维度调整 [C, T, H, W] -> [T, H, W, C]
    video = video.permute(1, 2, 3, 0)
    
    # 3. 保存
    logger.info(f"   Saving to {save_path}...")
    torchvision.io.write_video(save_path, video, fps=fps)

def main():
    config_path = "config/stylization.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    logger.info(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    
    # 初始化 Options
    opt = Options(
        root=cfg.dataset.root,
        index=cfg.dataset.index,
        clip_len=cfg.dataset.clip_len,
        stride=cfg.dataset.stride,
        height=cfg.dataset.height,
        width=cfg.dataset.width,
        batch_size=cfg.dataset.batch_size,
        num_workers=4, 
        phase="train"
    )
    
    # 初始化数据集
    logger.info("Initializing Dataset...")
    dataset = Style1000DatasetWrapper(opt)
    logger.info(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # 初始化 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True, 
        num_workers=opt.num_workers
    )
    
    # 获取一个 Batch
    logger.info("Fetching a batch...")
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        logger.error(f"Failed to fetch batch: {e}")
        return

    # 检查内容
    source = batch['source_video']
    target = batch['target_video']
    prompts = batch['prompt']
    video_ids = batch['video_id']
    
    logger.info("=" * 40)
    logger.info(f"Batch Size: {source.shape[0]}")
    logger.info(f"Source Shape: {source.shape}")
    logger.info(f"Data Type: {source.dtype}")
    # 此时预期应该是 [0, 255]
    logger.info(f"Value Range: [{source.min():.1f}, {source.max():.1f}]") 
    logger.info("=" * 40)

    # 保存可视化结果
    output_dir = "debug_vis"
    os.makedirs(output_dir, exist_ok=True)
    
    idx = 0 
    vid_id = video_ids[idx]
    
    # 打印 Prompt 方便检查
    print(f"\n>>>> 🟢 CHECK PROMPT <<<<")
    print(f"Video ID: {vid_id}")
    print(f"Prompt: {prompts[idx]}")
    print(f">>>> ---------------- <<<<\n")
    
    # 保存 Source (原视频)
    src_path = os.path.join(output_dir, f"vis_source_id{vid_id}.mp4")
    save_video_smart(source[idx], src_path)
    
    # 保存 Target (风格化参考)
    tgt_path = os.path.join(output_dir, f"vis_target_id{vid_id}.mp4")
    save_video_smart(target[idx], tgt_path)
    
    logger.info("=" * 40)
    logger.info(f"Check '{output_dir}' for visualized videos.")

if __name__ == "__main__":
    main()