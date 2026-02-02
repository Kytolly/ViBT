import os
import sys
import torch
import argparse
from torchvision.io import write_video

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from vibt.env import CONFIG
from vibt.wan import WanModel
from vibt.inference import generate_vibt
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint folder (containing adapter_model.bin)")
    parser.add_argument("--video_path", type=str, required=True, help="Input ego video path")
    parser.add_argument("--prompt", type=str, default="Transform ego view to third-person view")
    parser.add_argument("--output", type=str, default="output.mp4")
    args = parser.parse_args()

    device = "cuda"
    
    # 1. 加载 Base Model
    print(f"🚀 Loading Base Model: {CONFIG.model.path}")
    dtype = torch.bfloat16 if CONFIG.training.mixed_precision == "bf16" else torch.float16
    base_model = WanModel.from_pretrained(CONFIG.model.path, device=device, dtype=dtype)
    
    # 2. 加载 LoRA
    print(f"🔄 Loading LoRA from: {args.ckpt_path}")
    # 注意: PeftModel.from_pretrained 需要作用于 transformer 组件
    base_model.transformer = PeftModel.from_pretrained(base_model.transformer, args.ckpt_path)
    base_model.eval()

    # 3. 执行推理
    print("🎨 Generating...")
    # 假设 CONFIG 中有尺寸配置，或者使用默认
    output_tensor = generate_vibt(
        base_model, 
        args.video_path, 
        args.prompt, 
        steps=50,
        target_size=(CONFIG.dataset.height, CONFIG.dataset.width),
        device=device
    )
    
    # 4. 保存视频
    # output_tensor: [1, C, F, H, W] range [-1, 1]
    vid = output_tensor[0].permute(1, 2, 3, 0) # [F, H, W, C]
    vid = (vid * 0.5 + 0.5).clamp(0, 1) * 255
    vid = vid.to(torch.uint8).cpu()
    
    write_video(args.output, vid, fps=8)
    print(f"✅ Saved to {args.output}")

if __name__ == "__main__":
    main()