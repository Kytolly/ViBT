import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# 将当前目录加入路径，确保能 import vibt
sys.path.append(os.getcwd())

try:
    from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("请确保脚本运行在项目根目录下，并且 vibt/ 文件夹中包含 __init__.py")
    sys.exit(1)

def check_paths(opt):
    """预检查路径是否存在，避免 Dataset 内部报错难以调试"""
    print(f"\n🔍 正在检查路径配置...")
    
    # 1. 检查 dataset 软链接
    if not os.path.exists(opt.assets):
        print(f"❌ 错误: 找不到数据集根目录 '{opt.assets}'。")
        print(f"   请确保你已经执行了: ln -s /path/to/real/data {opt.assets}")
        return False
    
    # 2. 检查 phase 目录 (例如 dataset/train)
    phase_root = os.path.join(opt.assets, opt.phase)
    if not os.path.exists(phase_root):
        print(f"❌ 错误: 找不到 Phase 目录 '{phase_root}'。")
        print(f"   请检查你的 FollowBench 数据集结构是否包含 '{opt.phase}' 文件夹。")
        return False

    # 3. 检查 index json
    index_path = os.path.join(phase_root, opt.annotation)
    if not os.path.exists(index_path):
        print(f"❌ 错误: 找不到索引文件 '{index_path}'。")
        print(f"   请确认 index.json 是否位于 '{phase_root}' 下。")
        return False
        
    print("✅ 路径检查通过！")
    return True

def denormalize(tensor):
    """将归一化的 Tensor (-1, 1) 还原为 (0, 1) 用于保存查看"""
    return tensor * 0.5 + 0.5

def main():
    # 1. 初始化配置
    opt = Options()
    
    # 根据你的实际情况调整这些参数
    opt.assets = "dataset"        # 你的软链接名称
    opt.phase = "train"           # 你的数据文件夹名 (train/test)
    opt.annotation = "index.json" # 你的索引文件名
    opt.height = 512              # 测试分辨率
    opt.width = 896               # 测试分辨率 (Wan2.1 推荐 16倍数)
    opt.clip_len = 16             # 测试帧数 (避免加载太慢)

    # 临时修复：如果你还没修改源码中的 bug，这就动态修复一下
    if not hasattr(opt, 'index'):
        opt.index = opt.annotation

    if not check_paths(opt):
        return

    print(f"\n🚀 开始初始化 Dataset...")
    try:
        dataset = FollowBenchDatasetWrapper(opt)
        print(f"✅ Dataset 初始化成功，共发现 {len(dataset)} 个样本。")
    except Exception as e:
        print(f"❌ Dataset 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 测试 DataLoader
    print(f"\n🔄 正在尝试读取第一个 Batch...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # num_workers=0 方便调试报错

    try:
        for i, batch in enumerate(dataloader):
            vid_id = batch['video_id'][0]
            ego_video = batch['ego_video']
            exo_video = batch['exo_video']
            ref_image = batch['ref_image']

            print(f"\n✅ 成功加载样本 ID: {vid_id}")
            print(f"   - Ego Video Shape: {ego_video.shape} (Expect: [B, T, C, H, W] or [B, C, T, H, W])")
            print(f"   - Exo Video Shape: {exo_video.shape}")
            print(f"   - Ref Image Shape: {ref_image.shape}")

            # 检查是否有全 0 数据 (意味着加载失败被 try-except 捕获并返回了 zeros)
            if torch.all(ego_video == 0):
                print("⚠️ 警告: Ego Video 全为 0，说明视频文件读取失败 (utils.load_video_to_device 出错)。")
                print("   请检查 utils.py 中的视频读取逻辑或文件路径是否正确。")
            
            # 保存一帧用于验证
            os.makedirs("debug_output", exist_ok=True)
            
            # 取第一帧 (假设形状是 [B, T, C, H, W])
            if ego_video.dim() == 5:
                frame_tensor = ego_video[0, 0] # T=0
            else:
                # 假设形状是 [B, C, T, H, W]
                frame_tensor = ego_video[0, :, 0] 

            save_path = f"debug_output/test_{vid_id}.png"
            save_image(denormalize(frame_tensor), save_path)
            print(f"📸 已保存测试帧到: {save_path} (请查看图像是否正常)")

            break # 只测试一个
            
    except Exception as e:
        print(f"❌ 读取数据时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()