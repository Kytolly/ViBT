import os
import sys
import torch
import torchvision
from torch.utils.data import DataLoader

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from vibt.env import CONFIG
from vibt.wan import WanModel
from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options

def main():
    print("🕵️‍♂️ Starting Detective Mode...")
    device = "cuda"
    dtype = torch.bfloat16
    
    # 1. 拿一个 Batch 的数据
    print("1. Loading Dataset...")
    opt = Options()
    # 强制覆盖配置，确保和训练一致
    opt.root = CONFIG.dataset.root
    opt.phase = CONFIG.dataset.phase
    opt.index = CONFIG.dataset.index
    opt.height = CONFIG.dataset.height
    opt.width = CONFIG.dataset.width
    opt.clip_len = CONFIG.dataset.clip_len
    opt.stride = CONFIG.dataset.stride
    
    dataset = FollowBenchDatasetWrapper(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        print(f"❌ DataLoader crashed: {e}")
        return

    ego = batch['ego_video']
    exo = batch['exo_video']
    
    # 2. 检查数值统计
    print(f"\n📊 Data Statistics:")
    print(f"   Ego Shape: {ego.shape}")
    print(f"   Ego Range: min={ego.min():.4f}, max={ego.max():.4f}, mean={ego.mean():.4f}")
    
    # 🚨 关键检查点：如果 min/max 都是 -1.0 或 0，说明读到的是黑帧！
    if ego.min() == -1.0 and ego.max() == -1.0:
        print("\n❌❌ CRITICAL ERROR: Ego video is pure BLACK/BLANK! Training is meaningless.")
        return
    if ego.min() == 0.0 and ego.max() == 0.0:
        print("\n❌❌ CRITICAL ERROR: Ego video is pure ZEROS! Training is meaningless.")
        return

    # 3. 检查 VAE 重建能力
    print("\n2. Testing VAE Reconstruction...")
    model = WanModel.from_pretrained(CONFIG.model.path, device=device, dtype=dtype)
    model.eval()
    
    ego = ego.to(device, dtype=dtype)
    
    with torch.no_grad():
        # Encode
        z = model.encode(ego)
        print(f"   Latent Shape: {z.shape}")
        print(f"   Latent Range: min={z.min():.4f}, max={z.max():.4f}")
        
        # Decode
        rec = model.decode(z)
        
    # 4. 保存对比图
    print("\n3. Saving Diagnosis Video...")
    # [B, C, T, H, W] -> [T, H, W, C]
    def to_vid(t):
        t = t[0].permute(1, 2, 3, 0).float()
        t = (t * 0.5 + 0.5).clamp(0, 1) * 255
        return t.to(torch.uint8).cpu()

    orig = to_vid(ego)
    recon = to_vid(rec)
    
    # 左右拼接
    combined = torch.cat([orig, recon], dim=2)
    torchvision.io.write_video("debug_data_vae.mp4", combined, fps=8)
    print(f"✅ Saved 'debug_data_vae.mp4'. Please check it immediately!")
    print("   Left: What the model sees (Input) | Right: What the model decodes (VAE)")

if __name__ == "__main__":
    main()