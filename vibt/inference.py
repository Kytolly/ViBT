import torch
import logging
from diffusers.utils import load_video

# 导入项目模块
from vibt.wan import WanModel, encode_video
from vibt.scheduler import ViBTScheduler

logger = logging.getLogger(__name__)

def generate_vibt(
    model: WanModel, 
    source_input, 
    prompt: str, 
    steps: int = 28,             # 作者 Notebook 默认使用 28 步
    target_size: tuple = None, 
    device: str = "cuda",
    shift_gamma: float = 5.0,    # 核心参数：时间位移
    noise_scale: float = 1.0,    # 核心参数：SDE 噪声强度
    guidance_scale: float = 1.5  # 核心参数：CFG 强度 (作者推荐 1.5)
):
    """
    执行 ViBT 推理：完全采用作者官方接口 (Pipeline + ViBTScheduler)
    """
    pipe = model.pipe
    
    # 1. 自动挂载 ViBTScheduler
    if not isinstance(pipe.scheduler, ViBTScheduler):
        logger.info("🔄 Swapping pipeline scheduler to ViBTScheduler...")
        # 从原有配置加载，保留兼容性
        pipe.scheduler = ViBTScheduler.from_config(pipe.scheduler.config)
    
    # 设置 ViBT 特有的参数: SDE (noise=1.0) + Time Shift (gamma=5.0)
    pipe.scheduler.set_parameters(noise_scale=noise_scale, shift_gamma=shift_gamma)

    # 2. 编码 Latents (必须使用确定性编码)
    # 作者的 encode_video 使用 posterior.mode()，而 WanModel.encode 使用 sample()
    # 推理时必须用 mode() 以保证稳定性
    latents = None
    
    if isinstance(source_input, str):
        logger.info(f"🎬 Loading video from path: {source_input}")
        # 使用 diffusers 工具加载视频 (返回 List[PIL.Image])
        video_frames = load_video(source_input)
        # 调用作者提供的 encode_video (vibt.wan)
        latents = encode_video(pipe, video_frames)
        
    elif isinstance(source_input, torch.Tensor):
        logger.info(f"🎬 Using Tensor input: {source_input.shape}")
        # 手动执行确定性编码 (复刻 encode_video 的逻辑)
        # 输入假设: [C, F, H, W] 或 [1, C, F, H, W], 范围 [-1, 1]
        with torch.no_grad():
            video_tensor = source_input.to(device=pipe.device, dtype=pipe.dtype)
            if video_tensor.dim() == 4:
                video_tensor = video_tensor.unsqueeze(0) # [1, C, F, H, W]
            
            # VAE Encode
            posterior = pipe.vae.encode(video_tensor, return_dict=False)[0]
            z = posterior.mode() # [关键] 使用 mode() 而非 sample()
            
            # Wan 特有的归一化处理
            latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
            latents = (z - latents_mean) * latents_std
            
    else:
        raise ValueError(f"Unsupported input type: {type(source_input)}")

    # 3. 执行 Pipeline 推理
    logger.info(f"🎨 Running Pipeline (Steps={steps}, CFG={guidance_scale}, Gamma={shift_gamma})...")
    
    # 调用 pipe (对应 video_stylization.ipynb 中的用法)
    # output_type="pt" 返回 Tensor [-1, 1]
    output = pipe(
        prompt=prompt,
        latents=latents,              # 传入 Ego Latents 作为起点
        num_inference_steps=steps,
        guidance_scale=guidance_scale,# 启用 CFG
        output_type="pt"              
    ).frames
    
    # 4. 格式对齐
    # Run_inference.py 期望 [1, C, F, H, W]
    # Pipeline 通常返回 [B, C, F, H, W] 或 [B, F, C, H, W]，视版本而定
    # WanPipeline (Diffusers版) 这里的 frames 通常是 [B, C, F, H, W]
    if output.ndim == 5:
        # 如果是 [B, F, C, H, W] 且 C=3，则 permute
        if output.shape[1] != 3 and output.shape[2] == 3:
            output = output.permute(0, 2, 1, 3, 4)
            
    return output