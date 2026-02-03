import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torchvision
import logging
logger = logging.getLogger(__name__)

# 导入项目模块
from vibt.wan import WanModel
from vibt.dataset_wrapper import FollowBenchDatasetWrapper, Options

class ViBTTrainer:
    def __init__(self, cfg):
        """
        Args:
            cfg (DictConfig): 由 vibt.env.CONFIG 提供的全局配置对象
        """
        self.cfg = cfg
        self.device = "cuda"
        
        # 1. 创建目录
        logger.info(f"📂 Creating directories: {self.cfg.project.output_dir}")
        os.makedirs(self.cfg.project.output_dir, exist_ok=True)
        os.makedirs(self.cfg.project.logging_dir, exist_ok=True)
        
        # 2. 初始化 WandB
        self._init_wandb()
        
        # 3. 加载模型
        logger.info(f"🚀 Loading WanModel from {self.cfg.model.path}...")
        dtype = torch.bfloat16 if self.cfg.training.mixed_precision == "bf16" else torch.float16
        
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
            if self.cfg.training.gradient_checkpointing:
                self.model.transformer.enable_gradient_checkpointing()
            
            for param in self.model.transformer.parameters():
                param.requires_grad = True
            params_to_optimize = self.model.transformer.parameters()
            
        # 5. 准备优化器
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.cfg.training.lr
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
        user_resume_path = getattr(self.cfg.training, "resume_path", "")
        
        if user_resume_path:
            # 策略 A: 用户明确指定了路径
            self._resume_training(user_resume_path)
        else:
            # 策略 B: 自动检测最新检查点
            latest_ckpt = self._find_latest_checkpoint(self.cfg.project.output_dir)
            if latest_ckpt:
                logger.info(f"✨ Auto-detected latest checkpoint: {latest_ckpt}")
                self._resume_training(latest_ckpt)
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
            name=f"vibt-lora-ep{self.cfg.training.epochs}",
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
        opt.assets = self.cfg.dataset.root
        opt.phase = self.cfg.dataset.phase
        opt.annotation = self.cfg.dataset.index
        opt.index = self.cfg.dataset.index
        
        opt.height = self.cfg.dataset.height
        opt.width = self.cfg.dataset.width
        opt.clip_len = self.cfg.dataset.clip_len
        opt.stride = self.cfg.dataset.stride
        
        logger.info(f"📚 Loading dataset from {opt.assets} ({opt.phase})...")
        dataset = FollowBenchDatasetWrapper(opt)
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
                
                if 'ego_video' not in batch:
                    raise ValueError(f"Batch missing keys. Found: {batch.keys()}")
                
                ego = batch['ego_video']
                if ego.dim() != 5:
                    raise ValueError(f"Incorrect tensor shape: {ego.shape}. Expected [B, C, T, H, W].")
                
                if torch.isnan(ego).any() or torch.isinf(ego).any():
                    raise ValueError("❌ NaN or Inf detected in 'ego_video'!")
                    
                if ego.min() == -1 and ego.max() == -1:
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
        self.model.transformer.eval() 

        ego_video = self.val_batch['ego_video'][:1].to(self.device) 
        exo_video_gt = self.val_batch['exo_video'][:1].to(self.device)
        
        prompt = self.cfg.training.instruction
        prompt_embeds = self.model.encode_prompt([prompt])
        
        # Encode
        z_curr = self.model.encode(ego_video)
        
        # Euler Sampling
        steps = 10 
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = i / steps
            t_input = torch.tensor([t_curr * 1000], device=self.device, dtype=z_curr.dtype)
            pred_v = self.model(z_curr, t_input, prompt_embeds)
            z_curr = z_curr + pred_v * dt
            
        pred_video = self.model.decode(z_curr) 
        
        # ---------------------------------------------------------
        # [修改] 三屏合一拼接保存逻辑 (Ego | Pred | GT)
        # ---------------------------------------------------------
        try:
            sample_dir = os.path.join(self.cfg.project.output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            save_path = os.path.join(sample_dir, f"val_{step_tag}_combined.mp4")

            # 1. 拼接: 在 Width 维度 (dim=4) 上拼接
            # [B, C, T, H, W] -> [B, C, T, H, W*3]
            combined_tensor = torch.cat([ego_video, pred_video, exo_video_gt], dim=4)

            # 2. 转换格式: [B, C, T, H, W] -> [T, H, W, C]
            local_vid = combined_tensor[0].permute(1, 2, 3, 0).float() 
            local_vid = (local_vid * 0.5 + 0.5).clamp(0, 1) # 反归一化
            local_vid = (local_vid * 255).to(torch.uint8)
            
            # 3. 保存
            torchvision.io.write_video(save_path, local_vid.cpu(), fps=8)
            logger.info(f"   💾 Saved COMBINED sample (Ego|Pred|GT) to {save_path}")
        except Exception as e:
            logger.error(f"   ❌ Failed to save local video: {e}")

        # WandB Logging
        def process_for_wandb(vid_tensor):
            vid = vid_tensor[0].permute(1, 0, 2, 3) 
            vid = (vid * 0.5 + 0.5).clamp(0, 1)     
            vid = (vid * 255).to(torch.uint8)
            return vid.cpu().numpy()

        try:
            wandb.log({
                "val/source_ego": wandb.Video(process_for_wandb(ego_video), fps=8, format="mp4"),
                "val/generated_exo": wandb.Video(process_for_wandb(pred_video), fps=8, format="mp4"),
                "val/ground_truth": wandb.Video(process_for_wandb(exo_video_gt), fps=8, format="mp4"),
                "global_step": self.global_step
            })
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")
        
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

    def compute_loss(self, ego_pixel, exo_pixel, prompts):
        prompt_embeds = self.model.encode_prompt(prompts)
        
        with torch.no_grad():
            z_0 = self.model.encode(ego_pixel) 
            z_1 = self.model.encode(exo_pixel) 
            
        B = z_0.shape[0]
        t = torch.rand((B,), device=self.device, dtype=z_0.dtype)
        t_expand = t.view(B, 1, 1, 1, 1)
        
        z_t = (1 - t_expand) * z_0 + t_expand * z_1
        target_v = z_1 - z_0
        
        t_input = t * 1000 
        
        pred_v = self.model(
            hidden_states=z_t,
            timestep=t_input,
            encoder_hidden_states=prompt_embeds
        )
        
        return F.mse_loss(pred_v, target_v)

    def train(self):
        self.model.transformer.train()
        total_epochs = self.cfg.training.epochs
        accum_steps = self.cfg.training.gradient_accumulation_steps
        
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
                    
                ego_video = batch['ego_video'].to(self.device)
                exo_video = batch['exo_video'].to(self.device)
                
                prompts = [self.cfg.training.instruction] * ego_video.shape[0]
                
                loss = self.compute_loss(ego_video, exo_video, prompts)
                loss = loss / accum_steps
                loss.backward()
                
                if (step + 1) % accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    if self.global_step % self.cfg.training.log_interval == 0:
                        wandb.log({
                            "train_loss": loss.item() * accum_steps, 
                            "lr": self.optimizer.param_groups[0]['lr'],
                            "epoch": epoch
                        })
                    
                    if self.global_step > 0 and self.global_step % self.cfg.training.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}", epoch=epoch)
                
                current_loss = loss.item() * accum_steps
                epoch_loss += current_loss
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            self.save_checkpoint(f"epoch_{epoch+1}", epoch=epoch+1)
            batches_to_skip = 0 
            logger.info(f"🏁 Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.dataloader):.4f}")