import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# -----------------------------------------------------------------------------
# 1. 环境设置：确保能导入项目模块
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../ViBT/unittest
project_root = os.path.dirname(current_dir)              # .../ViBT
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # 从 env.py 导入全局配置 (已包含绝对路径)
    from vibt.env import CONFIG
    # 导入数据集和选项类
    from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("请确保在项目根目录下运行，且 vibt 包结构完整。")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 2. 工具函数
# -----------------------------------------------------------------------------
def denormalize(tensor):
    """
    反归一化：将 [-1, 1] 的 Tensor 映射回 [0, 1] 以便可视化。
    注意：您之前修改了 utils.py，现在的视频数据范围是 [-1, 1]。
    """
    return tensor * 0.5 + 0.5

# -----------------------------------------------------------------------------
# 3. 主测试逻辑
# -----------------------------------------------------------------------------
def main():
    print(f"🚀 [Test] Starting Dataset Loading Test...")
    print(f"   Config File: {os.getenv('CONFIG_PATH', 'Default (config/video2video.yml)')}")

    # --- A. 配置映射 ---
    # 将 env.py 中的 OmegaConf 配置映射到 dataset_wrapper.py 需要的 Options 对象
    opt = Options()
    
    # 路径参数 (env.py 已经处理成了绝对路径)
    opt.assets = CONFIG.dataset.root
    opt.phase = CONFIG.dataset.phase
    opt.annotation = CONFIG.dataset.index
    
    # 兼容性处理：防止 dataset_wrapper 内部使用 .index 而非 .annotation
    # (根据之前的代码分析，dataset_wrapper._load_index 可能使用了 self.opt.index)
    opt.index = CONFIG.dataset.index

    # 尺寸参数
    opt.height = CONFIG.dataset.height
    opt.width = CONFIG.dataset.width
    opt.clip_len = 16  # 测试时只读取 16 帧以加快速度
    
    print(f"\n📋 Configuration:")
    print(f"   Root Path:   {opt.assets}")
    print(f"   Index File:  {opt.annotation}")
    print(f"   Phase:       {opt.phase}")
    print(f"   Resolution:  {opt.height}x{opt.width}")

    # --- B. 初始化数据集 ---
    print(f"\n🔄 Initializing Dataset...")
    try:
        dataset = FollowBenchDatasetWrapper(opt)
        print(f"✅ Dataset initialized successfully.")
        print(f"   Total Samples: {len(dataset)}")
    except Exception as e:
        print(f"❌ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(dataset) == 0:
        print("⚠️ Warning: Dataset is empty! Please check your index.json and paths.")
        return

    # --- C. 测试 DataLoader 读取 ---
    print(f"\n🔄 Fetching first batch...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # num_workers=0 方便调试

    try:
        batch = next(iter(loader))
        
        # 提取数据
        vid_id = batch['video_id'][0]
        ego_video = batch['ego_video'] # Expect: [B, T, C, H, W] due to stack
        exo_video = batch['exo_video']
        ref_image = batch['ref_image'] # Expect: [B, C, H, W]

        print(f"✅ Batch loaded successfully!")
        print(f"   Video ID:    {vid_id}")
        print(f"   Ego Shape:   {ego_video.shape}")
        print(f"   Exo Shape:   {exo_video.shape}")
        print(f"   Ref Shape:   {ref_image.shape}")

        # 检查数值范围 (简单抽样)
        print(f"   Ego Range:   [{ego_video.min():.3f}, {ego_video.max():.3f}] (Expected approx [-1, 1])")
        
        # --- D. 保存可视化结果 ---
        os.makedirs("debug_output", exist_ok=True)
        save_path = f"debug_output/test_{vid_id}.png"
        
        # 取第一帧进行保存
        # DataLoader 堆叠后形状通常是 [B, T, C, H, W]
        # 我们取 Batch 0, Frame 0 -> [C, H, W]
        if ego_video.dim() == 5:
            # 假设 utils.load_video_to_device 返回 [T, C, H, W]
            frame_tensor = ego_video[0, 0] 
        else:
            # 备用情况
            frame_tensor = ego_video[0]

        # 反归一化并保存
        save_image(denormalize(frame_tensor), save_path)
        print(f"\n📸 Saved test frame to: {save_path}")
        print(f"   请检查图片：画面应清晰、色彩正常 (不应泛白或全黑)。")

    except Exception as e:
        print(f"❌ Error during batch fetching: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()