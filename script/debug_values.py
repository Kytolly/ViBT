import os
import sys
import torch
from safetensors.torch import load_file

# 环境设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from vibt.env import CONFIG
from vibt.wan import WanModel
from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options

def stat(name, tensor):
    return f"{name}: shape={list(tensor.shape)}, min={tensor.min():.2f}, max={tensor.max():.2f}, mean={tensor.mean():.2f}, std={tensor.std():.2f}"

def main():
    print("🔬 Deep Value Inspection...")
    device = "cuda"
    dtype = torch.bfloat16
    
    # 1. 加载模型
    model = WanModel.from_pretrained(CONFIG.model.path, device=device, dtype=dtype)
    model.eval()

    # 2. 加载 1700 步权重
    ckpt_path = os.path.join(CONFIG.project.output_dir, "checkpoint_step_1700") 
    # 自动寻找最新
    if not os.path.exists(ckpt_path):
        all_ckpts = [d for d in os.listdir(CONFIG.project.output_dir) if "checkpoint_step" in d]
        if all_ckpts:
            latest = sorted(all_ckpts, key=lambda x: int(x.split("_")[-1]))[-1]
            ckpt_path = os.path.join(CONFIG.project.output_dir, latest)
            
    print(f"🚀 Checkpoint: {ckpt_path}")
    weight_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
    if os.path.exists(weight_path):
        model.transformer.load_state_dict(load_file(weight_path), strict=False)
        print("✅ Weights loaded.")
    else:
        print("❌ No weights found.")
        return

    # 3. 准备一条训练数据
    opt = Options()
    opt.root = CONFIG.dataset.root; opt.phase = CONFIG.dataset.phase; opt.index = CONFIG.dataset.index
    opt.height = CONFIG.dataset.height; opt.width = CONFIG.dataset.width
    opt.clip_len = CONFIG.dataset.clip_len; opt.stride = CONFIG.dataset.stride
    dataset = FollowBenchDatasetWrapper(opt)
    batch = dataset[0]
    
    ego = batch['ego_video'].unsqueeze(0).to(device, dtype=dtype)
    gt = batch['exo_video'].unsqueeze(0).to(device, dtype=dtype)
    
    # 4. 模拟推理第一步 (t=0)
    print("\n-------------------------------------------------")
    print("🕵️‍♂️  Step 0 Inference Diagnostics")
    print("-------------------------------------------------")
    
    with torch.no_grad():
        # Encode
        z_0 = model.encode(ego)
        print(stat("Input z_0 (Ego Latent)", z_0))
        
        # Prepare Inputs
        prompt = CONFIG.training.instruction
        prompt_embeds = model.encode_prompt([prompt])
        t_input = torch.tensor([0.0], device=device, dtype=dtype) # t=0
        
        # Model Forward
        print("Running Model Forward at t=0...")
        pred_v = model(z_0, t_input, prompt_embeds)
        print(stat("Output pred_v (Velocity)", pred_v))
        
        # Calculate Expected v
        z_1 = model.encode(gt)
        target_v = z_1 - z_0
        print(stat("Target v (GT - Ego)", target_v))
        
        # Calculate Error
        mse = torch.nn.functional.mse_loss(pred_v, target_v)
        print(f"\n📉 Instant MSE Loss at t=0: {mse.item():.4f}")
        
        # Update z (Euler Step)
        dt = 1.0 / 20
        z_new = z_0 + pred_v * dt
        print(stat("Next z (z_0 + v*dt)", z_new))
        
        # Decode
        vid_new = model.decode(z_new)
        print(stat("Decoded Video Pixel", vid_new))

if __name__ == "__main__":
    main()