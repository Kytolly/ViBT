import os
import torch
import logging
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

# 导入您的项目模块
from vibt.dataset_wrapper import Style1000DatasetWrapper
from vibt.wan import WanModel
from vibt.env import CONFIG_STYLIZATION
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShapeChecker")

def check_dataloader_shapes():
    # 1. 加载配置
    cfg = CONFIG_STYLIZATION
    
    # 2. 初始化数据集
    logger.info(f"📂 正在从 {cfg.dataset.root} 加载数据集...")
    dataset = Style1000DatasetWrapper(cfg.dataset)
    
    # 3. 初始化 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 4. 预期形状计算
    # Wan2.1 视频张量标准形状: [Batch, Channel, Frames, Height, Width]
    expected_frames = cfg.dataset.clip_len
    expected_height = cfg.dataset.height
    expected_width = cfg.dataset.width
    
    logger.info(f"🎯 预期形状: [B, 3, {expected_frames}, {expected_height}, {expected_width}]")
    logger.info("🚀 开始遍历数据集...")

    error_count = 0
    for i, batch in enumerate(tqdm(dataloader)):
        source_video = batch['source_video']
        target_video = batch['target_video']
        
        # 检查 source_video 和 target_video 形状是否一致
        if source_video.shape != target_video.shape:
            logger.error(f"❌ Batch {i} 内部不匹配: Source {list(source_video.shape)} vs Target {list(target_video.shape)}")
            error_count += 1
            continue

        # 检查是否符合配置定义的 H, W, T
        # dim 0: Batch, dim 1: Channel(3), dim 2: Frames, dim 3: Height, dim 4: Width
        actual_frames = source_video.shape[2]
        actual_height = source_video.shape[3]
        actual_width = source_video.shape[4]

        if (actual_frames != expected_frames or 
            actual_height != expected_height or 
            actual_width != expected_width):
            
            logger.error(f"❌ Batch {i} 与配置不符!")
            logger.error(f"   实际: T={actual_frames}, H={actual_height}, W={actual_width}")
            logger.error(f"   配置: T={expected_frames}, H={expected_height}, W={expected_width}")
            error_count += 1
        
        # 针对您之前的 29 vs 30 报错进行专项检查
        # 如果 Width (dim 4) 不是预期值，记录下来
        if actual_width % 16 != 0:
            logger.warning(f"⚠️ Batch {i} 的宽度 {actual_width} 不是 16 的倍数，这可能导致 VAE 编解码后形状改变。")

    if error_count == 0:
        logger.info("🎉 [通过] 数据集 Dataloader 输出形状完全符合预期！")
    else:
        logger.error(f"🚨 [失败] 发现 {error_count} 处形状异常，请检查预处理 Resize 逻辑。")

if __name__ == "__main__":
    check_dataloader_shapes()