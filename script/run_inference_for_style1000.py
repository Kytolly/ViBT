import os
import sys
import argparse
import yaml
import torch
import logging
from torchvision.io import read_video, write_video
from torchvision import transforms
from peft import PeftModel
from safetensors.torch import load_file

# 假设你的代码包名为 vibt，请确保在 PYTHONPATH 中或在项目根目录运行
from vibt.wan import WanModel
from vibt.inference import generate_vibt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_video(video_path, height, width, device, dtype):
    """
    读取视频并预处理为 [1, C, F, H, W] 的 Tensor，归一化到 [-1, 1]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # 读取视频 [T, H, W, C] in [0, 255]
    vframes, _, _ = read_video(video_path, output_format="TCHW")
    
    # 调整维度 [T, C, H, W]
    if vframes.shape[-1] == 3: 
        vframes = vframes.permute(0, 3, 1, 2)
        
    # 强制 Resize
    resize = transforms.Resize((height, width))
    vframes = resize(vframes)
    
    # Normalize [0, 255] -> [-1, 1]
    vframes = vframes.to(device, dtype=dtype)
    vframes = vframes / 127.5 - 1.0
    
    # Add Batch Dim: [1, C, T, H, W]
    # 注意：WanModel 的 VAE 编码通常需要 [B, C, T, H, W] 或者是 [B, C, F, H, W] 
    # generate_vibt 内部如果处理的是 Latent 逻辑，我们需要看 inference.py。
    # 通常 VAE encode 接受 [B, C, F, H, W]
    video_tensor = vframes.permute(1, 0, 2, 3).unsqueeze(0)
    
    return video_tensor

def main():
    parser = argparse.ArgumentParser(description="Run inference for ViBT (LoRA or Full FT)")
    
    # 核心路径参数
    parser.add_argument("--config", type=str, default="config/stylization.yaml", help="Path to experiment config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint folder (e.g., outputs/xxx/checkpoint_epoch_5)")
    parser.add_argument("--input_video", type=str, required=True, help="Path to source video")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output_path", type=str, default="inference_result.mp4", help="Path to save output video")
    
    # 可选覆盖参数 (如果不传则使用 config 中的默认值)
    parser.add_argument("--noise_scale", type=float, default=None, help="Overide noise_scale (e.g., 0.0 for ODE)")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--num_steps", type=int, default=None, help="Inference steps")
    parser.add_argument("--use_lora", action="store_true", help="Force use LoRA (override config)")
    parser.add_argument("--full_ft", action="store_true", help="Force use Full FT (override config)")
    
    args = parser.parse_args()
    
    # 1. 加载配置
    logger.info(f"📂 Loading config from {args.config}")
    cfg = load_yaml_config(args.config)
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # 2. 确定模型类型 (LoRA vs Full)
    use_lora = cfg['model'].get('use_lora', False)
    if args.use_lora: use_lora = True
    if args.full_ft: use_lora = False
    
    logger.info(f"🏗️ Model Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'}")
    
    # 3. 加载底模 (Base Model)
    base_model_path = cfg['model']['path']
    logger.info(f"🚀 Loading Base WanModel from {base_model_path}...")
    model = WanModel.from_pretrained(base_model_path, device=device, dtype=dtype)
    
    # 4. 加载训练权重 (Checkpoint)
    ckpt_path = args.checkpoint
    logger.info(f"♻️ Loading Checkpoint from {ckpt_path}...")
    
    if use_lora:
        # === LoRA 加载逻辑 ===
        # 检查是否是 PEFT 格式
        if os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
            model.transformer = PeftModel.from_pretrained(model.transformer, ckpt_path)
            model.transformer.merge_and_unload() # 可选：合并权重以加速推理
        else:
            raise ValueError(f"Checkpoint {ckpt_path} does not contain adapter_config.json. Is this a LoRA checkpoint?")
    else:
        # === 全量微调加载逻辑 ===
        # 通常全量微调保存的是 diffusion_pytorch_model.safetensors 或 .bin
        # 或者是 transformer.save_pretrained 的结果
        
        # 尝试查找 safetensors
        safetensors_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(ckpt_path, "diffusion_pytorch_model.bin")
        
        state_dict = None
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            # 可能是直接保存的 transformer 文件夹结构
            try:
                from transformers import  WanTransformer3DModel # 假设底层是这个，或者直接用 load_state_dict
                # 但这里我们最好直接用 load_state_dict，因为 model.transformer 已经初始化了
                # 如果 checkpont 是通过 save_pretrained 保存的，目录里应该有 config.json
                pass
            except:
                pass
        
        if state_dict is not None:
            # 处理可能的 key 前缀不匹配 (比如 DDP 训练留下的 module. 前缀)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."): k = k[7:]
                new_state_dict[k] = v
            
            missing, unexpected = model.transformer.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            # 尝试使用 save_pretrained 的加载方式
            logger.info("Trying to load using transformers native from_pretrained logic on top of base...")
            # 注意：这可能会重新初始化，比较慢，最好是有 state_dict
            try:
                # 这是一个假设的 API，具体取决于 WanTransformer 的实现
                # 如果是 Diffusers 的 ModelMixin:
                from diffusers.models import ModelMixin
                if isinstance(model.transformer, ModelMixin):
                    loaded_transformer = model.transformer.__class__.from_pretrained(ckpt_path, torch_dtype=dtype)
                    model.transformer = loaded_transformer.to(device)
            except Exception as e:
                raise ValueError(f"Could not load full weights from {ckpt_path}. Error: {e}")

    model.eval()
    
    # 5. 准备输入数据
    height = cfg['dataset']['height']
    width = cfg['dataset']['width']
    
    logger.info(f"🎞️ Preprocessing video {args.input_video} ({height}x{width})...")
    source_tensor = preprocess_video(args.input_video, height, width, device, dtype)
    
    # 6. 推理参数
    inf_cfg = cfg['inference']
    steps = args.num_steps if args.num_steps else inf_cfg['num_inference_steps']
    noise_scale = args.noise_scale if args.noise_scale is not None else inf_cfg['noise_scale']
    shift_gamma = inf_cfg['shift_gamma']
    seed = inf_cfg['seed']
    
    logger.info(f"🧪 Inference Config: Steps={steps}, Noise={noise_scale}, CFG={args.guidance_scale}, Seed={seed}")
    
    # 7. 执行生成
    # 注意：generate_vibt 需要在 inference.py 中定义，且支持 Tensor 输入
    generated_latents = generate_vibt(
        model=model,
        source_input=source_tensor, # [1, C, F, H, W] in [-1, 1]
        prompt=args.prompt,
        steps=steps,
        noise_scale=noise_scale,
        shift_gamma=shift_gamma,
        guidance_scale=args.guidance_scale,
        seed=seed,
        device=device
    )
    
    # 8. 解码与保存
    logger.info("🎨 Decoding and saving...")
    # 假设 generated_latents 已经是 decode 好的视频或者 inference 内部做了 decode
    # 查看 inference.py 的 generate_vibt 通常返回的是 tensor 还是 video path?
    # 如果 generate_vibt 返回的是 [B, C, F, H, W] 的 Tensor (Pixel Space or Latent?)
    # 根据之前的 train.py 逻辑，inference 过程通常返回 Latent，需要解码。
    # 但如果是 generate_vibt 封装好的函数，它可能直接返回 pixel tensor。
    # 我们假设它返回的是 Pixel Tensor [1, C, F, H, W] in [-1, 1]
    
    if isinstance(generated_latents, torch.Tensor):
        output_vid = generated_latents[0].permute(1, 2, 3, 0).float().cpu() # [F, H, W, C]
        output_vid = (output_vid * 0.5 + 0.5).clamp(0, 1) * 255
        output_vid = output_vid.to(torch.uint8)
        
        write_video(args.output_path, output_vid, fps=15)
        logger.info(f"💾 Saved result to {args.output_path}")
    else:
        logger.info("generate_vibt returned non-tensor, assuming it handled saving or return type is different.")

if __name__ == "__main__":
    '''
    python script/run_inference_for_style1000.py \
    --checkpoint "outputs/stylization-v2/checkpoint_step_26000" \
    --input_video "assets/1890934837-1-destyle.mp4" \
    --prompt "A grand medieval castle perches atop a rocky cliff, its stone walls and turrets bathed in the warm glow of a setting sun. The sky transitions from deep orange near the horizon to a darker blue above, with scattered clouds reflecting the sunlight. Below, a bustling harbor teems with activity as numerous wooden ships, some with sails furled, line the water. Figures in period-appropriate attire stand on the decks, suggesting a gathering or event. A sturdy stone bridge with arched supports connects the castle to the mainland, adorned with flags fluttering in the breeze. The overall scene captures a moment of historical significance, blending natural beauty." \
    --noise_scale 1.0  \
    --guidance_scale 1.0 \
    --num_steps 28 \
    --use_lora
    '''
    main()
