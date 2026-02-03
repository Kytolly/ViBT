import os
import sys
import torch
import torchvision
from safetensors.torch import load_file
import logging

# 环境设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from vibt.env import CONFIG
from vibt.wan import WanModel
from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options

def main():
    print("⚖️  Starting STRICT Weight Check...")
    device = "cuda"
    dtype = torch.bfloat16
    
    # 1. 准备模型
    print("1. Loading Base Model...")
    model = WanModel.from_pretrained(CONFIG.model.path, device=device, dtype=dtype)
    model.eval()

    # 2. 锁定检查点
    ckpt_path = os.path.join(CONFIG.project.output_dir, "checkpoint_step_1700") # <--- 请确认您的步数
    if not os.path.exists(ckpt_path):
        # 尝试找最新的
        latest = [d for d in os.listdir(CONFIG.project.output_dir) if "checkpoint_step" in d]
        if latest:
            latest.sort(key=lambda x: int(x.split("_")[-1]))
            ckpt_path = os.path.join(CONFIG.project.output_dir, latest[-1])
    
    print(f"🚀 Target Checkpoint: {ckpt_path}")
    weight_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
    
    if not os.path.exists(weight_path):
        print(f"❌ Weight file not found: {weight_path}")
        return

    # 3. 加载权重并打印详细报告
    print("2. Loading State Dict...")
    state_dict = load_file(weight_path)
    
    # [关键诊断] 检查 Keys
    ckpt_keys = list(state_dict.keys())
    model_keys = list(model.transformer.state_dict().keys())
    
    print(f"   Checkpoint contains {len(ckpt_keys)} keys.")
    print(f"   Model expects {len(model_keys)} keys.")
    print(f"   Sample Ckpt Key:  {ckpt_keys[0]}")
    print(f"   Sample Model Key: {model_keys[0]}")
    
    # 尝试加载
    print("3. Attempting load_state_dict...")
    # 注意：这里我们检查 model.transformer
    missing, unexpected = model.transformer.load_state_dict(state_dict, strict=False)
    
    print("\n📊 Loading Report:")
    print(f"   ❌ Missing Keys: {len(missing)}")
    print(f"   ❓ Unexpected Keys: {len(unexpected)}")
    
    if len(missing) > 0:
        print(f"   Example Missing: {missing[0]}")
        # 如果 Missing 数量 > 0 且接近 len(model_keys)，说明加载完全失败！
        if len(missing) > len(model_keys) * 0.9:
            print("\n🚨 CRITICAL: Almost ALL keys are missing! Weight loading FAILED.")
            print("💡 Hint: Check if keys have 'module.' or 'transformer.' prefix mismatches.")
            
            # 尝试自动修复前缀
            print("🛠️  Attempting Auto-Fix for Prefixes...")
            new_state_dict = {}
            for k, v in state_dict.items():
                # 常见修复策略
                new_k = k.replace("transformer.", "") # 去掉 transformer. 前缀
                new_state_dict[new_k] = v
            
            m2, u2 = model.transformer.load_state_dict(new_state_dict, strict=False)
            print(f"   [Retry] Missing: {len(m2)}, Unexpected: {len(u2)}")
            if len(m2) < len(missing):
                print("   ✅ Auto-Fix worked! Using fixed weights.")
            else:
                print("   ❌ Auto-Fix failed.")

    # 4. 执行推理 (训练集回测 Sanity Check)
    # 我们直接拿训练集的数据跑，因为 Loss 0.19 意味着它在训练集上一定表现很好。
    # 如果这里跑出来还是噪声，那就彻底说明是权重问题。
    print("\n4. Running Sanity Inference on TRAINING Data...")
    
    opt = Options()
    opt.root = CONFIG.dataset.root; opt.phase = CONFIG.dataset.phase; opt.index = CONFIG.dataset.index
    opt.height = CONFIG.dataset.height; opt.width = CONFIG.dataset.width
    opt.clip_len = CONFIG.dataset.clip_len; opt.stride = CONFIG.dataset.stride
    dataset = FollowBenchDatasetWrapper(opt)
    
    # 取第0个样本 (假设它参与了训练)
    batch = dataset[0]
    ego = batch['ego_video'].unsqueeze(0).to(device, dtype=dtype)
    gt = batch['exo_video'].unsqueeze(0).to(device, dtype=dtype)
    
    with torch.no_grad():
        prompt_embeds = model.encode_prompt([CONFIG.training.instruction])
        z = model.encode(ego)
        
        # 简单采样
        steps = 20
        dt = 1.0 / steps
        for i in range(steps):
            t_input = torch.tensor([i/steps * 1000], device=device, dtype=dtype)
            v = model(z, t_input, prompt_embeds)
            z = z + v * dt
        
        pred = model.decode(z)
        
    # 保存
    combined = torch.cat([ego, pred, gt], dim=4)
    local_vid = combined[0].permute(1, 2, 3, 0).float()
    local_vid = (local_vid * 0.5 + 0.5).clamp(0, 1) * 255
    torchvision.io.write_video("eval_sanity_check.mp4", local_vid.to(torch.uint8).cpu(), fps=8)
    print(f"\n✅ Saved 'eval_sanity_check.mp4'.")
    print("   Look at the MIDDLE video. If it's noise, weights are definitely not loaded.")

if __name__ == "__main__":
    main()