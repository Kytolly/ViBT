import torch
import os
import torchvision
from diffusers.utils import export_to_video

# 导入项目模块
from vibt.utils import load_video_to_device
from vibt.wan import WanModel

def generate_vibt(
    model: WanModel, 
    source_video_path: str, 
    prompt: str, 
    steps: int = 50,
    target_size: tuple = None, # (H, W)
    device: str = "cuda"
):
    """
    执行 ViBT 推理：Ego Video -> Exo Video
    """
    model.eval()
    
    # 1. 预处理源视频 (关键修正)
    # 必须使用 vibt.utils 加载，以确保归一化范围是 [-1, 1]
    # 如果使用 torchvision 原生读取，范围不一致会导致生成结果异常
    print(f"🎬 Loading source video: {source_video_path}")
    source_pixel = load_video_to_device(
        source_video_path, 
        target_size=target_size, 
        device=device
    )
    
    if source_pixel is None:
        raise ValueError(f"Failed to load video from {source_video_path}")
    
    # 增加 Batch 维度: [C, F, H, W] -> [1, C, F, H, W]
    source_pixel = source_pixel.unsqueeze(0)

    # 2. 准备 Prompt Embedding
    prompt_embeds = model.encode_prompt([prompt])

    with torch.no_grad():
        # 3. 编码源视频 -> z_0 (Start State)
        # model.encode 会处理 [-1, 1] 到 Latent 的转换
        z_curr = model.encode(source_pixel)
        
        print(f"🔄 Starting Euler integration ({steps} steps)...")
        
        # 4. Euler 积分循环 (Bridge Matching Trajectory)
        # 从 t=0 (Ego) -> t=1 (Exo)
        dt = 1.0 / steps
        
        for i in range(steps):
            t_curr = i / steps
            
            # 构造时间步输入 (Wan2.1 训练时时间步范围通常是 0-1000)
            t_input = torch.tensor([t_curr * 1000], device=device, dtype=z_curr.dtype)
            
            # 预测速度 v = Model(z_t, t, prompt)
            pred_v = model(
                hidden_states=z_curr, 
                timestep=t_input, 
                encoder_hidden_states=prompt_embeds
            )
            
            # 更新状态: z_{t+dt} = z_t + v * dt
            z_curr = z_curr + pred_v * dt
            
        # 5. 解码 -> z_1 (End State)
        print("Decode latents...")
        output_pixel = model.decode(z_curr)
        
    return output_pixel