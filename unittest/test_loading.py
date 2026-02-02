import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 环境设置 Hack
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../unittest
project_root = os.path.dirname(current_dir)              # .../ViBT
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from vibt.env import CONFIG
    from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 2. 辅助工具
# -----------------------------------------------------------------------------
def denormalize(tensor):
    """[-1, 1] -> [0, 1]"""
    return tensor * 0.5 + 0.5

def check_tensor(name, tensor, expected_range=(-1, 1)):
    """检查 Tensor 的数值健康状况"""
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    status = "✅"
    if has_nan or has_inf:
        status = "❌ NAN/INF"
    elif min_val < expected_range[0] - 0.1 or max_val > expected_range[1] + 0.1:
        status = "⚠️ Range Warn"
        
    print(f"   - {name}: Shape={list(tensor.shape)} | Range=[{min_val:.3f}, {max_val:.3f}] | {status}")
    return not (has_nan or has_inf)

# -----------------------------------------------------------------------------
# 3. 核心测试逻辑
# -----------------------------------------------------------------------------
def main():
    print(f"==================================================")
    print(f"   ViBT Dataset Detailed Inspection")
    print(f"==================================================")
    
    # 1. 打印配置
    print(f"📋 Configuration:")
    print(f"   Root:       {CONFIG.dataset.root}")
    print(f"   Index:      {CONFIG.dataset.index}")
    print(f"   Clip Len:   {CONFIG.dataset.clip_len} (Target Frames)")
    print(f"   Resolution: {CONFIG.dataset.height}x{CONFIG.dataset.width}")
    
    # 2. 初始化 Dataset
    print(f"\n🔄 Initializing Dataset...")
    opt = Options()
    # 映射配置
    opt.assets = CONFIG.dataset.root
    opt.phase = CONFIG.dataset.phase
    opt.annotation = CONFIG.dataset.index
    opt.index = CONFIG.dataset.index
    opt.height = CONFIG.dataset.height
    opt.width = CONFIG.dataset.width
    opt.clip_len = CONFIG.dataset.clip_len
    # 强制设置用于测试的参数
    opt.batch_size = 2 # 关键：测试 Batch 堆叠
    
    try:
        dataset = FollowBenchDatasetWrapper(opt)
        print(f"✅ Dataset initialized. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"❌ Dataset Init Failed: {e}")
        return

    if len(dataset) == 0:
        print("❌ Dataset is empty.")
        return

    # 3. 单样本检查 (Single Sample Inspection)
    print(f"\n🔍 [Check 1] Single Sample Inspection (Index 0)...")
    try:
        sample = dataset[0]
        vid_id = sample['video_id']
        ego = sample['ego_video']
        exo = sample['exo_video']
        
        print(f"   ID: {vid_id}")
        check_tensor("Ego Video", ego)
        check_tensor("Exo Video", exo)
        
        # 验证帧数是否等于 clip_len
        if ego.shape[0] != opt.clip_len:
            print(f"   ❌ Frame count mismatch! Expected {opt.clip_len}, got {ego.shape[0]}")
        else:
            print(f"   ✅ Frame count matches clip_len ({opt.clip_len}).")
            
    except Exception as e:
        print(f"❌ Single sample load failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Batch Collate 检查 (DataLoader Test)
    print(f"\nmic [Check 2] DataLoader Batch Collate Test (Batch Size = 2)...")
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, # Shuffle to mix different length videos
        num_workers=0 # Debug mode
    )
    
    try:
        batch = next(iter(dataloader))
        print("✅ DataLoader successfully collated a batch!")
        
        ego_batch = batch['ego_video'] # Expect [B, T, C, H, W]
        print(f"   Batch Shape: {ego_batch.shape}")
        
        if ego_batch.shape[0] != 2:
            print(f"   ⚠️ Expected batch size 2, got {ego_batch.shape[0]} (Maybe dataset too small?)")
            
        # 5. 可视化检查
        print(f"\n🖼️ [Check 3] Visual Inspection...")
        os.makedirs("debug_output", exist_ok=True)
        
        # 取第一个样本的第一帧
        # Layout: [B, T, C, H, W] -> [C, H, W]
        frame0_tensor = ego_batch[0, 0] 
        save_path = "debug_output/debug_frame_start.png"
        save_image(denormalize(frame0_tensor), save_path)
        print(f"   Saved start frame to: {save_path}")
        
        # 取第一个样本的最后一帧 (检查是否黑屏或异常)
        frame_end_tensor = ego_batch[0, -1]
        save_path_end = "debug_output/debug_frame_end.png"
        save_image(denormalize(frame_end_tensor), save_path_end)
        print(f"   Saved end frame to:   {save_path_end}")
        
        print("\n✅ All Checks Passed! You are ready to train.")
        
    except RuntimeError as e:
        print(f"❌ DataLoader Collate Failed: {e}")
        print("   Hint: This usually means video frames are not unified. Did you apply the 'max_frames' fix in utils.py?")
    except Exception as e:
        print(f"❌ Unknown Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()