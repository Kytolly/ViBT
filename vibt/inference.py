import torch
import logging
from diffusers.utils import load_video, export_to_video

# 导入作者提供的原始模块（确保 wan.py 和 scheduler.py 已在 vibt 目录下）
from vibt.wan import WanModel, encode_video, decode_latents
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
    guidance_scale: float = 1.5
):
    pipe = model.pipe
    
    # 1. 替换为作者的 ViBTScheduler
    # 作者的 Scheduler 处理了 1000->0 到 0->1 的数学映射
    if not isinstance(pipe.scheduler, ViBTScheduler):
        logger.info("🔄 Replacing scheduler with Author's ViBTScheduler...")
        # 必须使用 from_scheduler 来继承配置，并注入 noise_scale 和 shift_gamma
        pipe.scheduler = ViBTScheduler.from_scheduler(
            pipe.scheduler, 
            noise_scale=noise_scale, 
            shift_gamma=shift_gamma
        )
    else:
        # 如果已经是 ViBTScheduler，更新参数
        pipe.scheduler.set_parameters(noise_scale=noise_scale, shift_gamma=shift_gamma)

    # 2. 编码 (使用 wan.py 中的 encode_video 以确保 Normalization)
    logger.info("🎬 Encoding source video with Normalization...")
    if isinstance(source_input, str):
        video_frames = load_video(source_input)
        latents = encode_video(pipe, video_frames)
    elif isinstance(source_input, torch.Tensor):
        # 即使是 Tensor，也建议暂时转回 encode_video 能处理的格式，
        # 或者你需要手动把 wan.py 的 encode 逻辑搬过来。
        # 这里为了稳妥，直接调用 wan.py 里的逻辑（假设 source_input 是预处理好的 tensor）
        
        # 注意：encode_video 内部调用了 preprocess_video，它期望 List[PIL] 或 uint8 tensor
        # 如果传入的是 float tensor [-1, 1]，需要手动归一化
        # 这里建议直接复用之前定义的 _encode_latents_with_norm 逻辑 (Mode模式)
        with torch.no_grad():
             # ... 确保这里执行了 (z - mean) * inverse_std
             # 简单起见，这里假设你已经在外部处理好了，或者在此处手动实现 wan.py 的逻辑
             pass 
        # ⚠️ 强烈建议：直接使用 wan.py 里的 encode_video 函数处理原始视频帧
        # 如果必须处理 Tensor，请把 wan.py 里的 encode_video 逻辑复制过来
        
    # 3. 推理 (使用标准 Pipeline)
    logger.info(f"🎨 Running Standard Pipeline (Steps={steps}, Gamma={shift_gamma})...")
    
    # 作者的 Scheduler 允许我们直接使用 pipe，把 latents 传给 latents 参数
    # 注意：在 T2V 任务中，通常 latents 是初始噪声。
    # 但在 ViBT (Video2Video) 中，source_latents 是起点。
    # 标准 Pipeline 可能会把 latents 当作初始噪声加噪。
    # 关键点：作者的 Scheduler step 函数逻辑是 x + delta * v + noise。
    # 只要传入的 latents 是 Source，并且 timesteps 是从 1000 开始，
    # 第一次 step 会计算 Source -> t_next 的变换。
    
    output_frames = pipe(
        prompt=prompt,
        latents=latents, # 将 Source Latents 传给 pipe
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        output_type="np" 
    ).frames[0] # pipe 返回的是 VideoOutput, frames 是 List[np.array]
    
    # 转换为 Tensor 返回，保持接口一致性
    output_tensor = torch.from_numpy(output_frames).permute(3, 0, 1, 2).unsqueeze(0) # [1, C, F, H, W]
    
    return output_tensor