import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
from torchvision import transforms
from PIL import Image
from decord import VideoReader, cpu
from safetensors.torch import load_file
from peft import PeftModel


# ==========================================
# 🔧 用户配置区 (请修改这里!)
# ==========================================
# 指向你的 checkpoint 文件夹 (例如 step_24000)
CHECKPOINT_PATH = "outputs/stylization-v2/checkpoint_step_24000"

# Style1000 数据集根目录
DATASET_ROOT = "/opt/liblibai-models/user-workspace2/datasets/style1000_aligned"

# index 文件路径 (通常在 dataset root 下或者单独存放)
INDEX_PATH = os.path.join(DATASET_ROOT, "index_for_train.json")

# 你的 Base Model 路径
BASE_MODEL_PATH = "/opt/liblibai-models/user-workspace2/model_zoo/Wan2.1-T2V-1.3B-Diffusers"

# 视频参数 (必须与训练时一致)
HEIGHT = 480
WIDTH = 832
CLIP_LEN = 29  
STRIDE = 4
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ViBT-Style1000")

def load_video_tensor(path, resize_func):
    """简化的视频读取函数，不依赖 dataset_wrapper"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    
    # 简单的取前 N 帧，如果不够这就循环
    indices = np.arange(0, CLIP_LEN * STRIDE, STRIDE)
    indices = np.clip(indices, 0, total_frames - 1)
    
    frames_np = vr.get_batch(indices).asnumpy() # [T, H, W, C]
    
    tensors = []
    for i in range(len(frames_np)):
        img = Image.fromarray(frames_np[i])
        img = resize_func(img)
        tensors.append(torch.from_numpy(np.array(img)))
    
    # [T, H, W, C] -> [C, T, H, W]
    vid_tensor = torch.stack(tensors).permute(3, 0, 1, 2).float()
    
    # Normalize to [-1, 1]
    vid_tensor = vid_tensor / 127.5 - 1.0
    return vid_tensor.unsqueeze(0) # [1, C, T, H, W]

def main():
    device = "cuda"
    dtype = torch.bfloat16
    
    # 1. 验证数据路径
    if not os.path.exists(INDEX_PATH):
        logger.error(f"❌ Index file not found at: {INDEX_PATH}")
        return
    
    logger.info("📂 Loading Index...")
    with open(INDEX_PATH, 'r') as f:
        data_entries = json.load(f)
    
    if len(data_entries) == 0:
        logger.error("❌ Index file is empty!")
        return
    
    # 随机取一个样本
    import random
    entry = random.choice(data_entries)
    logger.info(f"🔍 Selected Sample ID: {entry.get('video_id', 'Unknown')}")
    
    # 拼接完整路径
    # Style1000 index 通常包含相对路径，如 "train/source/xxx.mp4"
    src_rel = entry['destyle'] # 注意：根据 dataset_wrapper，destyle 是 source
    tgt_rel = entry['style']   # style 是 target
    
    src_path = os.path.join(DATASET_ROOT, src_rel)
    tgt_path = os.path.join(DATASET_ROOT, tgt_rel)
    prompt_text = entry.get('caption', "style transfer")
    
    logger.info(f"   Source: {src_path}")
    logger.info(f"   Target: {tgt_path}")
    logger.info(f"   Prompt: {prompt_text}")

    # 2. 加载模型
    logger.info("🚀 Loading Model...")
    from vibt.wan import WanModel
    model = WanModel.from_pretrained(BASE_MODEL_PATH, device=device, dtype=dtype)
    
    # 加载权重
    if os.path.isdir(CHECKPOINT_PATH):
        # 优先找 Lora
        if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_model.bin")):
            logger.info("   Found LoRA adapter.")
            model.transformer = PeftModel.from_pretrained(model.transformer, CHECKPOINT_PATH)
        # 其次找 Full Finetune
        elif os.path.exists(os.path.join(CHECKPOINT_PATH, "diffusion_pytorch_model.safetensors")):
            logger.info("   Found Full Weights (safetensors).")
            full_w = load_file(os.path.join(CHECKPOINT_PATH, "diffusion_pytorch_model.safetensors"))
            model.transformer.load_state_dict(full_w, strict=False)
        else:
            logger.warning("⚠️ No recognized weights found in checkpoint folder! Using Base Model only.")
    
    model.eval()
    model.transformer.to(device) # 确保 transformer 在 GPU
    
    # 3. 处理数据
    logger.info("🎞️ Processing Video & Encoding Latents...")
    resize = transforms.Resize((HEIGHT, WIDTH))
    
    try:
        x0_pixel = load_video_tensor(src_path, resize).to(device, dtype=dtype)
        x1_pixel = load_video_tensor(tgt_path, resize).to(device, dtype=dtype)
    except Exception as e:
        logger.error(f"❌ Failed to load video: {e}")
        return

    with torch.no_grad():
        # VAE Encode
        # Wan VAE 期望: [B, C, F, H, W] -> Encode -> Latent
        # 注意：需要确认 VAE 的输入形状，如果是 [B, C, T, H, W]
        
        def encode_and_norm(x):
            # Encode
            dist = model.vae.encode(x).latent_dist
            z = dist.mode()
            # Normalize: (z - mean) / std
            if hasattr(model.vae.config, "latents_mean"):
                mean = torch.tensor(model.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
                std = torch.tensor(model.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
                z = (z - mean) / std
            return z
            
        z0 = encode_and_norm(x0_pixel)
        z1 = encode_and_norm(x1_pixel)
        
        prompt_emb = model.encode_prompt([prompt_text])
        
    logger.info(f"   Latent Shape: {z0.shape}")

    # 4. 诊断：预测方向 vs 真实方向
    logger.info("\n🧪 === DIAGNOSIS REPORT ===")
    
    # 真实方向 (Ground Truth Velocity)
    v_gt = z1 - z0
    
    # 让模型预测 t=0 (Source) 时的速度
    # 物理时间 t=0 -> Transformer Time T=1000
    t_input = torch.tensor([1000], device=device, dtype=dtype)
    
    with torch.no_grad():
        # Forward
        v_pred = model(z0, t_input, prompt_emb)
    
    # 计算指标
    v_pred_flat = v_pred.flatten().float()
    v_gt_flat = v_gt.flatten().float()
    
    # 1. 余弦相似度
    cos_sim = F.cosine_similarity(v_pred_flat, v_gt_flat, dim=0)
    
    # 2. 模长比率
    norm_pred = v_pred_flat.norm()
    norm_gt = v_gt_flat.norm()
    ratio = norm_pred / (norm_gt + 1e-6)
    
    logger.info(f"1. Direction (Cosine Sim): {cos_sim.item():.5f}")
    if cos_sim.item() > 0.05:
        logger.info("   ✅ PASS: Positive correlation. Model has learned the direction.")
    elif cos_sim.item() < -0.05:
        logger.warning("   ⚠️ FAIL: Negative correlation. Model is pushing AWAY from target.")
    else:
        logger.error("   ❌ FAIL: Near zero correlation. Model prediction is orthogonal/random.")
        
    logger.info(f"2. Magnitude (Pred/GT):    {ratio.item():.5f}")
    logger.info(f"   Pred Norm: {norm_pred:.4f}")
    logger.info(f"   GT Norm:   {norm_gt:.4f}")
    
    if ratio < 0.01:
        logger.warning("   ⚠️ WARNING: Prediction is too weak (Vanishing). Check Normalization.")
        
    # 5. 诊断：Scheduler 步进方向
    logger.info("\n🛠️ === SCHEDULER CHECK ===")
    from vibt.scheduler import ViBTScheduler
    scheduler = ViBTScheduler(num_train_timesteps=1000)
    scheduler.set_parameters(noise_scale=0.0, shift_gamma=5.0)
    scheduler.set_timesteps(20, device=device)
    
    # Step 1: T=1000 -> T=950
    timestep = scheduler.timesteps[0]
    output = scheduler.step(v_pred, timestep, z0)
    z_next = output[0]
    
    dist_old = (z0 - z1).norm()
    dist_new = (z_next - z1).norm()
    diff = dist_old - dist_new
    
    logger.info(f"Distance to Target (Start): {dist_old:.4f}")
    logger.info(f"Distance to Target (After): {dist_new:.4f}")
    logger.info(f"Improvement: {diff:.4f}")
    
    if diff > 0:
        logger.info("✅ PASS: Scheduler moved latent CLOSER to target.")
    else:
        logger.error("❌ FAIL: Scheduler moved latent FURTHER/SAME. Logic inverted?")

if __name__ == "__main__":
    main()