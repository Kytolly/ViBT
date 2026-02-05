import torch
import logging
import numpy as np
from diffusers.utils import load_video, export_to_video
from diffusers import WanPipeline

from vibt.wan import WanModel, encode_video
from vibt.scheduler import ViBTScheduler

logger = logging.getLogger(__name__)

def generate_vibt(
    model: WanModel, 
    source_input, 
    prompt: str, 
    steps: int = 28,
    target_size: tuple = None, 
    device: str = "cuda",
    shift_gamma: float = 5.0,
    noise_scale: float = 1.0,
    guidance_scale: float = 1.5,
    seed: int = 42
):
    """
    运行 ViBT 推理。
    
    Args:
        model: WanModel 实例
        source_input: 视频路径 (str) 或 预处理后的 Tensor ([-1, 1], Shape [B, C, F, H, W])
        prompt: 提示词
        ...
    """
    pipe: WanPipeline = model.pipe
    
    # =======================================================
    # 1. 准备 Scheduler (使用作者原生的 Backward Scheduler)
    # =======================================================
    if not isinstance(pipe.scheduler, ViBTScheduler):
        logger.info("🔄 Replacing scheduler with Author's ViBTScheduler...")
        pipe.scheduler = ViBTScheduler.from_scheduler(pipe.scheduler)
        pipe.scheduler.set_parameters(noise_scale=noise_scale, shift_gamma=shift_gamma, seed=seed)

    # =======================================================
    # 2. 准备 Latents (Source) 并进行 Normalization
    # =======================================================
    # 训练时模型见过的是 Normalized Latents，推理必须保持一致
    
    if isinstance(source_input, str):
        logger.info(f"🎬 Loading video from {source_input}...")
        video_frames = load_video(source_input)
        latents = encode_video(pipe, video_frames)
        
    elif isinstance(source_input, torch.Tensor):
        logger.info("🎬 Encoding tensor input (Validation)...")
        with torch.no_grad():
            pixel_values = source_input.to(device=device, dtype=model.vae.dtype)
            
            # 1. VAE Encode
            dist = model.vae.encode(pixel_values).latent_dist # [B, C, F, H, W]
            z = dist.mode() # 确定性
            
            # 2. Normalization
            # 必须与训练时的分布对齐
            vae_config = model.vae.config
            
            # 获取均值和方差参数
            # 注意: 作者代码中 latents_std 实际上是 1/std (inverse std)
            # vibt/wan.py: latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std)
            
            if hasattr(vae_config, "latents_mean"):
                latents_mean = torch.tensor(vae_config.latents_mean).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
                # 计算 1/std
                latents_std_inv = 1.0 / torch.tensor(vae_config.latents_std).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
                
                # 执行归一化: (z - mean) / std
                latents = (z - latents_mean) * latents_std_inv
            else:
                latents = z # Fallback
                
    # =======================================================
    # 3. 执行推理 (Standard Pipeline Call)
    # =======================================================
    # 此时 latents 代表 Source ($T=1000$ 对应的状态)
    # Pipeline 会从 Scheduler 获取 timesteps (1000 -> 0)
    
    logger.info(f"🎨 Running Inference (Backward 1000->0, Steps={steps})...")
    
    # 扩展 Prompt 以适配 Batch Size
    # 如果 latents 是 [B, C, F, H, W]，prompt 需要是 list 长度 B
    batch_size = latents.shape[0]
    if isinstance(prompt, str):
        prompts = [prompt] * batch_size
    else:
        prompts = prompt

    output_frames = pipe(
        prompt=prompts,
        latents=latents,         # 将 Source Latents 作为初始状态传入
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        output_type="np"         # 返回 numpy [B, F, H, W, C] in [0, 1]
    ).frames
    
    # =======================================================
    # 4. 格式转换 (适配 Trainer)
    # =======================================================
    # Trainer 期望: Tensor [B, C, F, H, W] in [-1, 1]
    # 当前 output_frames: List of List of np.array (如果 batch>1) 或 List of np.array
    
    # pipe 返回的是 VideoOutput, frames 是 List[np.ndarray] (每个元素是一个视频)
    # np.ndarray shape: [F, H, W, C]
    
    tensor_list = []
    for vid_np in output_frames:
        # np [F, H, W, C] -> Tensor [C, F, H, W]
        vid_tensor = torch.from_numpy(vid_np).permute(3, 0, 1, 2).float()
        tensor_list.append(vid_tensor)
        
    output_tensor = torch.stack(tensor_list).to(device) # [B, C, F, H, W]
    
    # [0, 1] -> [-1, 1]
    output_tensor = (output_tensor * 2.0) - 1.0
    
    return output_tensor