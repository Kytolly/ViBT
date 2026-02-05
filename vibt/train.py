import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torchvision
from omegaconf import OmegaConf, DictConfig
import logging
logger = logging.getLogger(__name__)

# 导入项目模块
from .wan import WanModel
from torch.utils.data import Dataset
from .dataset_wrapper import FollowBenchDatasetWrapper, Options, Style1000DatasetWrapper
from .scheduler import ViBTScheduler
from .env import ViBTEnvConfig
from .inference import generate_vibt

class ViBTTrainer:
    def __init__(self, cfg: ViBTEnvConfig, dataset: Dataset):
        """
        Args:
            cfg (ViBTEnvConfig): 由 vibt.env.ViBTEnvConfig 提供的全局配置对象
        """
        self.cfg = cfg
        self.device = "cuda"
        self.dataset = dataset
        
        # 1. 创建目录
        logger.info(f"📂 Creating directories: {self.cfg.project.output_dir}")
        os.makedirs(self.cfg.project.output_dir, exist_ok=True)
        logger.info(f"📂 Creating directories: {self.cfg.project.logging_dir}")
        os.makedirs(self.cfg.project.logging_dir, exist_ok=True)
        
        # 2. 初始化 WandB
        self._init_wandb()
        
        # 3. 加载模型
        logger.info(f"🚀 Loading WanModel from {self.cfg.model.path}...")
        dtype = torch.bfloat16 if self.cfg.train.mixed_precision == "bf16" else torch.float16
        
        self.model = WanModel.from_pretrained(
            self.cfg.model.path, 
            device=self.device,
            dtype=dtype
        )
        
        # 4. 配置训练模式 (LoRA vs Full)
        if self.cfg.model.use_lora:
            self._setup_lora()
            params_to_optimize = filter(lambda p: p.requires_grad, self.model.transformer.parameters())
        else:
            logger.info("🔓 Unfreezing Transformer for Full-Parameter Training...")
            if self.cfg.train.gradient_checkpointing:
                self.model.transformer.enable_gradient_checkpointing()
            
            for param in self.model.transformer.parameters():
                param.requires_grad = True
            params_to_optimize = self.model.transformer.parameters()
            
        # 5. 准备优化器
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.cfg.train.lr
        )
        
        # 6. 准备数据
        self._setup_dataloader()
        
        # 7. 健全性检查
        self._perform_sanity_check(num_steps=2)

        # 8. 准备固定验证样本
        self.val_batch = self._get_fixed_validation_batch()
        
        self.global_step = 0
        self.start_epoch = 0

        # 9. 智能恢复训练逻辑
        user_resume_path = self.cfg.train.resume_path
        if user_resume_path:
            self._resume_training(user_resume_path)
        else:
            logger.info("🆕 No checkpoint found. Starting fresh training.")

    def _init_wandb(self):
        id_file = os.path.join(self.cfg.project.output_dir, "wandb_id.txt")
        wandb_id = None
        resume_mode = "allow"
        
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                wandb_id = f.read().strip()

        run = wandb.init(
            project=self.cfg.project.name,
            dir=self.cfg.project.logging_dir,
            config=self._flatten_config(self.cfg),
            name=f"vibt-{self.cfg.project.name}-ep{self.cfg.train.epochs}",
            resume=resume_mode,
            id=wandb_id
        )
        
        if not os.path.exists(id_file):
            with open(id_file, 'w') as f:
                f.write(run.id)

    def _flatten_config(self, cfg):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)

    def _setup_lora(self):
        logger.info(f"💉 Injecting LoRA adapters (Rank={self.cfg.model.lora_rank})...")
        lora_config = LoraConfig(
            r=self.cfg.model.lora_rank,
            lora_alpha=self.cfg.model.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            bias="none"
        )
        self.model.transformer = get_peft_model(self.model.transformer, lora_config)
        self.model.transformer.print_trainable_parameters()

    def _setup_dataloader(self):
        opt = Options()
        opt.root = self.cfg.dataset.root
        opt.phase = self.cfg.dataset.phase
        opt.index = self.cfg.dataset.index
        
        opt.height = self.cfg.dataset.height
        opt.width = self.cfg.dataset.width
        opt.clip_len = self.cfg.dataset.clip_len
        opt.stride = self.cfg.dataset.stride
        
        logger.info(f"📚 Loading dataset from {opt.root} ({opt.phase})...")
        dataset = self.dataset
        logger.info(f"✅ Dataset loaded: {len(dataset)} samples.")
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

    def _perform_sanity_check(self, num_steps=2):
        """启动前深度体检"""
        logger.info(f"🩺 Performing dataloader sanity check ({num_steps} steps)...")
        
        lock_file = os.path.join(self.cfg.dataset.root, self.cfg.dataset.phase, "dataset_verified.lock")
        if os.path.exists(lock_file):
            logger.info("   🔒 [Cache] Found dataset_verified.lock. Skipping extensive checks.")
            return

        try:
            check_iter = iter(self.dataloader)
            for i in range(num_steps):
                logger.info(f"   [Sanity Check] Fetching batch {i+1}/{num_steps}...")
                batch = next(check_iter)
                
                if 'source_video' not in batch:
                    raise ValueError(f"Batch missing keys. Found: {batch.keys()}")
                
                source = batch['source_video']
                if source.dim() != 5:
                    raise ValueError(f"Incorrect tensor shape: {source.shape}. Expected [B, C, T, H, W].")
                
                if torch.isnan(source).any() or torch.isinf(source).any():
                    raise ValueError("❌ NaN or Inf detected in 'source_video'!")
                    
                if source.min() == -1 and source.max() == -1:
                    logger.warning(f"⚠️ Warning: Batch {i} contains purely black frames.")

            logger.info("   ✅ Sanity check passed!")
        except Exception as e:
            logger.error(f"   ❌ Sanity check failed: {e}")
            sys.exit(1)

    def _get_fixed_validation_batch(self):
        logger.info("🖼️ Fetching a fixed validation batch...")
        try:
            batch = next(iter(self.dataloader))
            return batch
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch validation batch: {e}")
            return None

    def _find_latest_checkpoint(self, output_dir):
        if not os.path.exists(output_dir): return None
        checkpoints = []
        for d in os.listdir(output_dir):
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(output_dir, d)):
                try:
                    step = int(d.split("_")[-1])
                    checkpoints.append((step, os.path.join(output_dir, d)))
                except ValueError: continue
        if not checkpoints: return None
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]

    def _resume_training(self, checkpoint_path):
        logger.info(f"🔄 Resuming training from {checkpoint_path}...")
        
        # 1. Weights
        full_weight_path = os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")
        if not self.cfg.model.use_lora and os.path.exists(full_weight_path):
            from safetensors.torch import load_file
            state_dict = load_file(full_weight_path)
            self.model.transformer.load_state_dict(state_dict, strict=False)
            logger.info("   ✅ Loaded Full Finetune weights.")
        elif self.cfg.model.use_lora:
            adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
            if os.path.exists(adapter_path):
                adapters_weights = torch.load(adapter_path, map_location=self.device)
                set_peft_model_state_dict(self.model.transformer, adapters_weights)
                logger.info("   ✅ Loaded LoRA weights.")
        
        # 2. Optimizer & State
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.start_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            logger.info(f"   ✅ Resumed from Epoch {self.start_epoch}, Step {self.global_step}.")

        # [新增] 3. 恢复后立即采样验证
        logger.info("🎨 Triggering validation sampling to verify resume status...")
        try:
            self.run_validation_sampling(f"resume_step_{self.global_step}")
        except Exception as e:
            logger.warning(f"⚠️ Resume sampling failed: {e}")

    @torch.no_grad()
    def run_validation_sampling(self, step_tag):
        if self.val_batch is None:
            return

        logger.info(f"🎨 Running validation sampling for {step_tag}...")
        
        # 1. 切换到评估模式 (Pipeline 内部也会处理，但显式调用更安全)
        self.model.transformer.eval() 
        
        # 2. 准备数据
        # 取 Batch 中的第一个样本进行验证
        source_video = self.val_batch['source_video'][:1] # [1, C, F, H, W]
        target_video_gt = self.val_batch['target_video'][:1].to(self.device)
        prompt = self.val_batch['prompt'][0] 
        
        try:
            pred_video = generate_vibt(
                model=self.model,
                source_input=source_video,  # 直接传入 Tensor
                prompt=prompt,
                steps=20,                # 验证时步数可以少一点以加快速度 (如 20)
                device=self.device,
                shift_gamma=5.0,         # 保持与论文一致
                noise_scale=1.0,         # SDE 模式
                guidance_scale=1.5       # 启用 CFG，这对于验证质量至关重要
            )
            
            # generate_vibt 返回的是 [1, C, F, H, W]，且已经在 [-1, 1] 范围内
            pred_video = pred_video.to(self.device)
            
            # 4. 拼接与保存 (三屏合一: Input | Pred | GT)
            sample_dir = os.path.join(self.cfg.project.output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            save_path = os.path.join(sample_dir, f"val_{step_tag}_combined.mp4")

            # 确保维度一致进行拼接
            combined_tensor = torch.cat([source_video.to(self.device), pred_video, target_video_gt], dim=4)
            
            # 格式转换 [C, F, H, W] -> [F, H, W, C]
            local_vid = combined_tensor[0].permute(1, 2, 3, 0).float() 
            local_vid = (local_vid * 0.5 + 0.5).clamp(0, 1) # Un-normalize
            local_vid = (local_vid * 255).to(torch.uint8)
            
            torchvision.io.write_video(save_path, local_vid.cpu(), fps=8)
            logger.info(f"   💾 Saved COMBINED sample to {save_path}")

            # 5. WandB Logging
            def process_for_wandb(vid_tensor):
                # WandB 需要 [T, C, H, W] 且值在 0-255 或 0-1
                # 输入: [1, C, F, H, W]
                vid = vid_tensor[0].permute(1, 0, 2, 3) 
                vid = (vid * 0.5 + 0.5).clamp(0, 1)     
                vid = (vid * 255).to(torch.uint8)
                return vid.cpu().numpy()

            wandb.log({
                "val/source": wandb.Video(process_for_wandb(source_video.to(self.device)), fps=8, format="mp4"),
                "val/generated": wandb.Video(process_for_wandb(pred_video), fps=8, format="mp4"),
                "val/ground_truth": wandb.Video(process_for_wandb(target_video_gt), fps=8, format="mp4"),
                "global_step": self.global_step
            })
            
        except Exception as e:
            logger.error(f"   ❌ Validation sampling failed: {e}")
            import traceback
            traceback.print_exc()

        # 6. 恢复训练模式
        self.model.transformer.train()

    def save_checkpoint(self, tag, epoch=None):
        path = os.path.join(self.cfg.project.output_dir, f"checkpoint_{tag}")
        os.makedirs(path, exist_ok=True)
        
        logger.info(f"💾 Saving checkpoint to {path}...")
        self.model.transformer.save_pretrained(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        
        current_epoch = epoch if epoch is not None else self.start_epoch
        torch.save({
            "epoch": current_epoch,
            "global_step": self.global_step
        }, os.path.join(path, "training_state.pt"))
        
        try:
            self.run_validation_sampling(tag)
        except Exception as e:
            logger.warning(f"⚠️ Validation sampling failed: {e}")
    
    def _encode_latents_with_norm(self, pixel_values):
        # 1. VAE Encode
        with torch.no_grad():
            pixel_values = pixel_values.to(dtype=self.model.vae.dtype, device=self.device)
            dist = self.model.vae.encode(pixel_values).latent_dist
            # 训练时通常采样 (sample)，推理用 mode
            z = dist.sample()
            
            # 2. [关键修复] Normalization
            # 获取 mean 和 std
            if not hasattr(self, 'latents_mean'):
                self.latents_mean = torch.tensor(self.model.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
                self.latents_std_inv = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
            
            # 将 mean/std 移动到正确设备（如果尚未移动）
            if self.latents_mean.device != z.device:
                self.latents_mean = self.latents_mean.to(z.device)
                self.latents_std_inv = self.latents_std_inv.to(z.device)

            # 执行标准化: (z - mean) / std
            latents = (z - self.latents_mean) * self.latents_std_inv
            
        return latents
    
    def compute_loss(self, ego_pixel, exo_pixel, prompts):
        # 0. 获取配置中的噪声缩放 s
        # 论文指出 s 需在训练和推理保持一致。
        # 你的配置文件中 s 定义在 inference.noise_scale，这里直接复用。
        s = self.cfg.inference.noise_scale 
        
        prompt_embeds = self.model.encode_prompt(prompts)
        
        with torch.no_grad():
            z_0 = self._encode_latents_with_norm(ego_pixel)  # Source
            z_1 = self._encode_latents_with_norm(exo_pixel)  # Target
            
        B = z_0.shape[0]
        # [关键] 获取总像素维度 D = C * F * H * W
        # 用于将 MSE 还原为 Sum of Squared Norm
        D = z_0[0].numel()
        
        # 1. 采样时间 t ~ U(0, 1)
        t = torch.rand((B,), device=self.device, dtype=z_0.dtype)
        
        # 2. 采样标准高斯噪声 epsilon ~ N(0, I)
        epsilon = torch.randn_like(z_0)
        t_expand = t.view(B, 1, 1, 1, 1)
        
        # 3. 构造中间状态 x_t (Algorithm S1, Step 3)
        # 公式: x_t = (1-t)x_0 + t*x_1 + s * sqrt(t(1-t)) * epsilon
        # [修改] 显式乘以 s
        bridge_noise_coeff = s * torch.sqrt(t_expand * (1 - t_expand))
        z_t = (1 - t_expand) * z_0 + t_expand * z_1 + bridge_noise_coeff * epsilon
        
        # 4. 计算目标速度 u_t (Algorithm S1, Step 4)
        # 公式: u_t = (x_1 - x_t) / (1-t)
        # 推导: u_t = (x_1 - x_0) - s * sqrt(t/(1-t)) * epsilon
        time_safe = torch.clamp(t_expand, min=1e-5, max=1.0 - 1e-5)
        
        # [修改] 显式乘以 s
        u_t_noise_coeff = s * torch.sqrt(time_safe / (1 - time_safe))
        target_v = (z_1 - z_0) - u_t_noise_coeff * epsilon
        
        # 5. 计算归一化因子 Alpha (Algorithm S1, Step 5)
        # 公式: alpha^2 = 1 + (s^2 * t * D) / ((1-t) * ||x_1 - x_0||^2)
        
        # 计算 ||x_1 - x_0||^2 (Sum of Squares)
        diff_norm_sq = torch.sum((z_1 - z_0) ** 2, dim=[1, 2, 3, 4]).view(B, 1, 1, 1, 1)
        diff_norm_sq = torch.clamp(diff_norm_sq, min=1e-6) # 防止除零
        
        # [修改] 分子部分乘以 s^2
        alpha_sq = 1 + (time_safe * D * (s**2)) / ((1 - time_safe) * diff_norm_sq)
        alpha = torch.sqrt(alpha_sq)
        
        # 6. 模型预测
        t_input = t * 1000 
        
        pred_v = self.model(
            hidden_states=z_t,
            timestep=t_input,
            encoder_hidden_states=prompt_embeds
        )

        # 7. 计算稳定速度匹配损失 (Algorithm S1, Step 6)
        # 论文目标: L = || (v - u) / alpha ||^2  (范数平方)
        # F.mse_loss 计算的是 Mean Squared Error (除以了 D)
        # 为了还原范数平方，需要乘以 D
        
        mse_loss = F.mse_loss(pred_v / alpha, target_v / alpha)
        loss = mse_loss * D
        
        return loss

    def train(self):
        self.model.transformer.train()
        total_epochs = self.cfg.train.epochs
        accum_steps = self.cfg.train.gradient_accumulation_steps
        
        # [核心] 计算断点续训需要跳过的步数
        batches_to_skip = 0
        if self.global_step > 0:
            batches_to_skip = (self.global_step * accum_steps) % len(self.dataloader)
            
        logger.info(f"🚀 Start Training from Epoch {self.start_epoch}...")
        if batches_to_skip > 0:
             logger.info(f"⏩ Resuming mid-epoch: Skipping first {batches_to_skip} batches to align with step {self.global_step}.")
        
        for epoch in range(self.start_epoch, total_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(pbar):
                # 跳过逻辑
                if epoch == self.start_epoch and step < batches_to_skip:
                    pbar.set_description(f"⏩ Skipping {step}/{batches_to_skip}")
                    continue
                elif pbar.desc.startswith("⏩"):
                    pbar.set_description(f"Epoch {epoch+1}/{total_epochs}")
                    
                source_video = batch['source_video'].to(self.device)
                target_video = batch['target_video'].to(self.device)
                prompt_batch = batch['prompt']
                
                prompts = [p for p in prompt_batch] * source_video.shape[0]
                
                loss = self.compute_loss(source_video, target_video, prompts)
                loss = loss / accum_steps
                loss.backward()
                
                if (step + 1) % accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    if self.global_step % self.cfg.train.log_interval == 0:
                        wandb.log({
                            "train_loss": loss.item() * accum_steps, 
                            "lr": self.optimizer.param_groups[0]['lr'],
                            "epoch": epoch
                        })
                    
                    if self.global_step > 0 and self.global_step % self.cfg.train.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}", epoch=epoch)
                
                current_loss = loss.item() * accum_steps
                epoch_loss += current_loss
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            self.save_checkpoint(f"epoch_{epoch+1}", epoch=epoch+1)
            batches_to_skip = 0 
            logger.info(f"🏁 Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.dataloader):.4f}")