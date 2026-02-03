import os
import sys
import torch
import torchvision
import logging
from torch.utils.data import DataLoader

# 1. 环境设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from vibt.env import CONFIG
from vibt.wan import WanModel
from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options
from peft import set_peft_model_state_dict, LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # === 配置区 ===
    checkpoint_path = os.path.join(CONFIG.project.output_dir, "checkpoint_step_1700") # <--- 指定检查点
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"🚀 Loading Checkpoint from: {checkpoint_path}")
    
    # 2. 加载模型
    model = WanModel.from_pretrained(CONFIG.model.path, device=device, dtype=dtype)
    
    # 3. 加载权重 (LoRA 或 Full)
    if CONFIG.model.use_lora:
        print("Loading LoRA adapters...")
        lora_config = LoraConfig(
            r=CONFIG.model.lora_rank, lora_alpha=CONFIG.model.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"], bias="none"
        )
        model.transformer = get_peft_model(model.transformer, lora_config)
        
        adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
        if os.path.exists(adapter_path):
            adapters = torch.load(adapter_path, map_location=device)
            set_peft_model_state_dict(model.transformer, adapters)
        else:
            print(f"❌ Adapter not found at {adapter_path}")
            return
    else:
        print("Loading Full Weights...")
        full_path = os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(full_path):
            from safetensors.torch import load_file
            state_dict = load_file(full_path)
            model.transformer.load_state_dict(state_dict, strict=False)
        else:
            print(f"❌ Weights not found at {full_path}")
            return

    model.eval()
    
    # 4. 准备数据 (只取一个 Batch)
    opt = Options()
    opt.root = CONFIG.dataset.root
    opt.phase = CONFIG.dataset.phase
    opt.index = CONFIG.dataset.index
    opt.height = CONFIG.dataset.height
    opt.width = CONFIG.dataset.width
    opt.clip_len = CONFIG.dataset.clip_len
    opt.stride = CONFIG.dataset.stride
    
    dataset = FollowBenchDatasetWrapper(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    
    ego_video = batch['ego_video'].to(device, dtype=dtype)
    exo_gt = batch['exo_video'].to(device, dtype=dtype)
    
    print("🎨 Running Inference...")
    
    # 5. 推理采样
    prompt = CONFIG.training.instruction
    prompt_embeds = model.encode_prompt([prompt])
    
    with torch.no_grad():
        z = model.encode(ego_video)
        
        # Euler Sampling
        steps = 20  # 稍微多一点步数以保证质量
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = i / steps
            t_input = torch.tensor([t_curr * 1000], device=device, dtype=dtype)
            pred_v = model(z, t_input, prompt_embeds)
            z = z + pred_v * dt
            
        pred_video = model.decode(z)

    # 6. 保存三屏对比视频
    # [B, C, T, H, W] -> Concat Width -> [T, H, W, C]
    combined = torch.cat([ego_video, pred_video, exo_gt], dim=4)
    local_vid = combined[0].permute(1, 2, 3, 0).float()
    local_vid = (local_vid * 0.5 + 0.5).clamp(0, 1)
    local_vid = (local_vid * 255).to(torch.uint8)
    
    save_path = "eval_result_step1700.mp4"
    torchvision.io.write_video(save_path, local_vid.cpu(), fps=8)
    print(f"✅ Saved result to {save_path}")

if __name__ == "__main__":
    main()