import torch
from vibt.wan import WanModel
from vibt.scheduler import FlowMatchScheduler # 假设 vibt 提供了这个或使用 diffusers
import torchvision.io

def generate_vibt(model, source_video_path, prompt, steps=50):
    device = "cuda"
    
    # 1. 预处理源视频
    # 读取视频并 resize/crop 到模型尺寸
    # ... (使用 torchvision.io.read_video)
    # source_pixel: [1, C, F, H, W]
    
    with torch.no_grad():
        # 2. 编码源视频 -> z_0
        z_curr = model.encode(source_pixel).to(device)
        
        # 3. Euler 积分循环 (从 t=0 到 t=1)
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = i / steps
            t_input = torch.tensor([t_curr * 1000], device=device) # 缩放时间步
            
            # 预测速度 v
            pred_v = model(z_curr, t_input, [prompt])
            
            # 更新状态: z_{t+dt} = z_t + v * dt
            z_curr = z_curr + pred_v * dt
            
        # 4. 解码 -> z_1
        output_pixel = model.decode(z_curr)
        
    return output_pixel

# 加载训练好的 LoRA
base_model = WanModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B")
from peft import PeftModel
model = PeftModel.from_pretrained(base_model.model, "output/vibt_epoch_9")
base_model.model = model # 替换回去

# 推理
out = generate_vibt(base_model, "test_ego.mp4", "Transform to third person view")