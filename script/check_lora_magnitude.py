import os
import sys
import torch
import numpy as np
from safetensors.torch import load_file

def check_lora_weights(ckpt_path):
    print(f"🔍 Analyzing checkpoint: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: File not found at {ckpt_path}")
        return

    try:
        # 加载权重
        state_dict = load_file(ckpt_path)
    except Exception as e:
        print(f"❌ Error loading safetensors: {e}")
        return

    print(f"📊 Total keys in adapter: {len(state_dict)}")
    
    lora_a_stats = []
    lora_b_stats = []
    
    for key, tensor in state_dict.items():
        tensor = tensor.float() # 转为 float32 计算统计量
        abs_mean = torch.mean(torch.abs(tensor)).item()
        max_val = torch.max(torch.abs(tensor)).item()
        
        # 分类统计 A 和 B 矩阵
        if "lora_A" in key or "down" in key: # down proj usually A
            lora_a_stats.append(abs_mean)
        elif "lora_B" in key or "up" in key: # up proj usually B (init with 0)
            lora_b_stats.append((key, abs_mean, max_val))
            
    # --- 分析结果 ---
    
    # 1. 检查 LoRA A (应该是随机初始化的，数值不应为0)
    if lora_a_stats:
        avg_a = np.mean(lora_a_stats)
        print(f"\n🔵 [LoRA A] (Initialization):")
        print(f"   Avg Abs Magnitude: {avg_a:.6f} (Should be roughly constant, ~0.03 for kaiming init)")
    
    # 2. 检查 LoRA B (初始化为0，反映学习程度)
    print(f"\n🔴 [LoRA B] (Learned Weights - Init is 0.0):")
    if not lora_b_stats:
        print("   No LoRA B keys found! (Check key names)")
        return

    # 计算 B 的整体统计
    b_means = [x[1] for x in lora_b_stats]
    b_maxs = [x[2] for x in lora_b_stats]
    
    avg_b = np.mean(b_means)
    max_b = np.max(b_maxs)
    
    print(f"   Avg Abs Magnitude: {avg_b:.8f}")
    print(f"   Max Value found:   {max_max_b:.8f}")
    
    # --- 诊断结论 ---
    print(f"\n🩺 Diagnosis:")
    if avg_b < 1e-5:
        print("❌ CRITICAL: LoRA B weights are extremely close to Zero!")
        print("   -> The model has learned ALMOST NOTHING.")
        print("   -> Cause: Likely the 'Loss Scaling' issue we fixed (gradients were ~0).")
    elif avg_b < 1e-3:
        print("⚠️ WARNING: LoRA B weights are very small.")
        print("   -> The model learned very slowly or learning rate was too low.")
    else:
        print("✅ HEALTHY: LoRA B weights show significant updates.")
        print("   -> The training dynamics seem to be working.")

    # 打印前几个具体的 B 权重供人工检查
    print("\n🔍 Sample LoRA B Keys:")
    for i in range(min(5, len(lora_b_stats))):
        k, mean, mx = lora_b_stats[i]
        print(f"   {k}: Mean={mean:.8f}, Max={mx:.8f}")

if __name__ == "__main__":
    # 指定你想要检查的文件路径
    ckpt_path = "outputs/stylization/checkpoint_step_18000/adapter_model.safetensors"
    
    # 支持命令行传参
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
        
    check_lora_weights(ckpt_path)