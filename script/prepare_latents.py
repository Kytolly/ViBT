import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# 确保能找到项目模块
sys.path.append(os.getcwd())

from vibt.wan import WanModel
from vibt.dataset_wrapper import Style1000DatasetWrapper, Options

def prepare_latents():
    # 1. 配置加载
    config_path = "config/stylization.yaml"
    cfg = OmegaConf.load(config_path)
    
    # 强制指定对齐后的数据集路径 (根据你的实际情况修改)
    # cfg.dataset.root = "/opt/liblibai-models/user-workspace2/datasets/style1000_aligned"
    # cfg.dataset.index = "index_for_train_aligned.json"

    # 输出目录
    output_dir = os.path.join(cfg.dataset.root, "latents_cache_v3_norm")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Loading VAE to GPU...")
    device = "cuda"
    # 加载模型 (dtype 使用 bfloat16 以节省空间且精度足够)
    model = WanModel.from_pretrained(cfg.model.path, device=device, dtype=torch.bfloat16)
    # 卸载 Transformer，预处理只需要 VAE
    model.transformer = None 
    torch.cuda.empty_cache()
    
    # 2. 准备数据集 (读取原始像素)
    opt = Options(
        root=cfg.dataset.root,
        index=cfg.dataset.index,
        clip_len=cfg.dataset.clip_len,
        stride=cfg.dataset.stride,
        height=cfg.dataset.height,
        width=cfg.dataset.width,
        batch_size=1,
        num_workers=8, # 多进程读取硬盘上的视频
        phase="train"
    )
    dataset = Style1000DatasetWrapper(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    
    print(f"📂 Saving latents to: {output_dir}")
    print(f"🔢 Total samples: {len(dataset)}")

    # 3. 预处理循环
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            video_id = batch['video_id'][0].item()
            save_path = os.path.join(output_dir, f"{video_id}.pt")
            
            if os.path.exists(save_path):
                continue
            
            # A. 像素归一化: uint8 [0, 255] -> bf16 [-1, 1]
            source = batch['source_video'].to(device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            target = batch['target_video'].to(device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            
            # B. VAE 编码 + Latent 归一化
            def process_to_latent(x):
                # Encode
                dist = model.vae.encode(x).latent_dist
                z = dist.mode() # 使用 Mode 保证确定性
                
                # Normalize (z - mean) / std
                if hasattr(model.vae.config, "latents_mean"):
                    mean = torch.tensor(model.vae.config.latents_mean).to(z)
                    std = torch.tensor(model.vae.config.latents_std).to(z)
                    z = (z - mean) / std
                return z.cpu() # 移回 CPU 准备保存

            z_source = process_to_latent(source)
            z_target = process_to_latent(target)
            prompt = batch['prompt'][0]
            
            # C. 保存
            torch.save({
                "source": z_source.squeeze(0), # [C, T, H, W]
                "target": z_target.squeeze(0),
                "prompt": prompt,
                "id": video_id
            }, save_path)

    print("✅ Done! VAE is no longer needed for training.")

if __name__ == "__main__":
    prepare_latents()