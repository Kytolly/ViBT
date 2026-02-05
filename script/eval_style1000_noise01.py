import os
import torch
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.utils import export_to_video

# 导入项目内部模块
from vibt.wan import WanModel
from vibt.dataset_wrapper import Style1000DatasetWrapper
from vibt.inference import generate_vibt

def run_eval():
    parser = argparse.ArgumentParser(description="ViBT Checkpoint Inference on Style1000")
    parser.add_argument("--ckpt_path", type=str, required=True, help="检查点文件夹路径 (如 output/checkpoint-15000)")
    parser.add_argument("--config", type=str, default="config/stylization.yaml", help="配置文件路径")
    parser.add_argument("--num_samples", type=int, default=5, help="验证样本数量")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = OmegaConf.load(args.config)
    
    # 2. 初始化底座模型
    print(f"🚀 Loading base model from {cfg.model.path}...")
    model = WanModel.from_pretrained(
        cfg.model.path,
        device="cuda",
        dtype=torch.bfloat16
    )
    
    # 3. 加载检查点权重 (LoRA 或全量)
    if cfg.model.use_lora:
        print(f"💉 Injecting LoRA weights from {args.ckpt_path}...")
        # 假设权重名为 pytorch_lora_weights.safetensors
        model.pipe.load_lora_weights(args.ckpt_path)
    else:
        print(f"🏗️ Loading full model weights from {args.ckpt_path}...")
        # 如果是全量微调，需要加载 Transformer 权重
        state_dict = torch.load(os.path.join(args.ckpt_path, "diffusion_pytorch_model.bin"), map_index="cpu")
        model.transformer.load_state_dict(state_dict)

    # 4. 准备 Style1000 数据集
    print(f"📂 Loading Style1000 dataset (Index: {cfg.dataset.index})...")
    dataset = Style1000DatasetWrapper(cfg.dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 5. 执行推理 (使用 noise_scale=0.1)
    output_dir = "eval_results_noise01"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎬 Starting inference with noise_scale=0.1...")
    model.transformer.eval()

    for i, batch in enumerate(dataloader):
        if i >= args.num_samples: break
        
        source_video = batch['source_video'].to("cuda") # destyle
        prompt = batch['prompt'][0]
        # video_id = os.path.basename(batch['video_id'][0]).split('.')[0]

        # print(f"  - Processing [{i+1}/{args.num_samples}]: {video_id}")
        
        with torch.no_grad():
            # 调用核心推理接口
            pred_video = generate_vibt(
                model=model,
                source_input=source_video,
                prompt=prompt,
                steps=20, 
                device="cuda",
                noise_scale=0.5,
                shift_gamma=5.0,
                guidance_scale=1.5
            )

        # 6. 保存结果
        save_path = os.path.join(output_dir, f"{i}_noise01.mp4")
        # pred_video 是 [B, C, T, H, W] Tensor
        video_tensor = pred_video[0].cpu().float() 
        
        # 2. 反归一化: [-1, 1] -> [0, 255]
        video_tensor = ((video_tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        
        # 3. 维度调整: [C, T, H, W] -> [T, H, W, C]
        video_np = video_tensor.permute(1, 2, 3, 0).numpy()
        
        # 4. 导出为视频 (转换为 list 确保稳定性)
        export_to_video(list(video_np), save_path, fps=24)
        
        print(f"  ✅ Saved to {save_path}")

if __name__ == "__main__":
    run_eval()