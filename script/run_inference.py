import os
import sys
import torch
import argparse
import random
from torchvision.io import write_video
from safetensors.torch import load_file
from peft import PeftModel

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from vibt.env import CONFIG
from vibt.wan import WanModel
from vibt.inference import generate_vibt
from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--video_path", type=str, default=None, help="Input video path (Optional)")
    parser.add_argument("--prompt", type=str, default="Transform ego view to third-person view")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--shift_gamma", type=float, default=5.0)
    # [新增] Dataset 模式参数
    parser.add_argument("--use_dataset", action="store_true", help="Load sample from dataset instead of file")
    parser.add_argument("--data_index", type=int, default=-1, help="Index of sample to load (-1 for random)")
    
    args = parser.parse_args()
    device = "cuda"
    
    # 1. 加载模型
    print(f"🚀 Loading Base Model...")
    dtype = torch.bfloat16 if CONFIG.training.mixed_precision == "bf16" else torch.float16
    base_model = WanModel.from_pretrained(CONFIG.model.path, device=device, dtype=dtype)
    
    # 2. 加载权重
    ckpt_path = args.ckpt_path
    if os.path.isfile(ckpt_path): ckpt_path = os.path.dirname(ckpt_path)
    
    full_weights = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
    lora_weights = os.path.join(ckpt_path, "adapter_model.bin")
    
    if os.path.exists(full_weights):
        print(f"🔄 Loading Full Finetune weights...")
        base_model.transformer.load_state_dict(load_file(full_weights), strict=False)
    elif os.path.exists(lora_weights):
        print(f"🔄 Loading LoRA adapter...")
        base_model.transformer = PeftModel.from_pretrained(base_model.transformer, ckpt_path)
    else:
        print(f"❌ No recognized weights found in {ckpt_path}")
        return
    base_model.eval()

    # 3. 准备输入数据 (Source)
    source_input = None
    
    if args.use_dataset:
        print("📚 Loading from DatasetWrapper (Robust Mode)...")
        opt = Options()
        opt.root = CONFIG.dataset.root; opt.phase = CONFIG.dataset.phase; opt.index = CONFIG.dataset.index
        opt.height = CONFIG.dataset.height; opt.width = CONFIG.dataset.width
        opt.clip_len = CONFIG.dataset.clip_len; opt.stride = CONFIG.dataset.stride
        
        dataset = FollowBenchDatasetWrapper(opt)
        
        # 尝试加载有效样本
        max_retries = 5
        for attempt in range(max_retries):
            try:
                idx = args.data_index if args.data_index >= 0 else random.randint(0, len(dataset)-1)
                print(f"   Trying dataset index: {idx} ...")
                batch = dataset[idx]
                
                # 拿到了！
                source_input = batch['source_video'] # [C, F, H, W]
                # 如果有 GT，也可以顺便拿出来对比
                gt_video = batch['target_video']
                print(f"   ✅ Successfully loaded index {idx}")
                break
            except Exception as e:
                print(f"   ⚠️ Failed to load index {idx}: {e}")
                if args.data_index >= 0: break # 指定了index就只试一次
        
        if source_input is None:
            print("❌ Failed to load any valid sample.")
            return
            
    else:
        # 文件模式
        if not args.video_path:
            print("❌ Error: Must provide either --video_path or --use_dataset")
            return
        source_input = args.video_path

    # 4. 执行推理
    print(f"🎨 Generating...")
    pred_tensor = generate_vibt(
        base_model, 
        source_input, 
        args.prompt, 
        steps=50,
        target_size=(CONFIG.dataset.height, CONFIG.dataset.width),
        device=device,
        shift_gamma=args.shift_gamma
    )
    
    # 5. 保存结果 (如果是Dataset模式，我们可以拼个GT做对比)
    # pred: [1, C, F, H, W]
    pred_vid = pred_tensor[0].permute(1, 2, 3, 0)
    
    # 归一化恢复
    pred_vid = (pred_vid * 0.5 + 0.5).clamp(0, 1) * 255
    
    # 如果有 GT，拼在一起
    if args.use_dataset and 'gt_video' in locals():
        gt_vid = gt_video.permute(1, 2, 3, 0) # [F, H, W, C]
        gt_vid = (gt_vid * 0.5 + 0.5).clamp(0, 1) * 255
        
        # 还要把 source 拼上
        src_vid = source_input.permute(1, 2, 3, 0).cpu()
        src_vid = (src_vid * 0.5 + 0.5).clamp(0, 1) * 255
        
        # 拼接: Ego | Pred | GT
        # 确保都在 CPU 和 uint8
        combined = torch.cat([src_vid, pred_vid.cpu(), gt_vid.cpu()], dim=2) # width维度拼接
        final_vid = combined.to(torch.uint8)
        print("   Layout: [Input Ego] | [Generated Exo] | [Ground Truth]")
    else:
        final_vid = pred_vid.to(torch.uint8).cpu()

    write_video(args.output, final_vid, fps=8)
    print(f"✅ Saved to {args.output}")

if __name__ == "__main__":
    main()