import os
import sys
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import prodigyopt
import logging

# 项目内引用
from .wan import WanModel
# 引用 dataset_wrapper 中的 JIT 逻辑和内存数据集
from .dataset_wrapper import Options, ensure_latents_cached, InMemoryLatentDataset
from .env import ViBTEnvConfig
from .inference import generate_vibt

# 日志配置
DETAILED_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
logging.basicConfig(level=logging.INFO, format=DETAILED_FORMAT, force=True)
logger = logging.getLogger(__name__)

class ViBTTrainer:
    def __init__(self, cfg: ViBTEnvConfig):
        self.cfg = cfg
        self.device = "cuda"
        self.avg_loss = None
        
        self._init_workspace()
        self._init_wandb()
        
        # 1. 显存优化版模型加载 (VAE Offload)
        self._setup_model_optimized()
        
        # 2. 优化器
        self._setup_optimizer()
        
        # 3. 自动化数据加载 (JIT Cache + RAM Dataset)
        self._setup_dataloader()
        
        # 4. 状态恢复
        self.val_batch = self._get_fixed_validation_batch()
        self.global_step = 0
        self.start_epoch = 0
        
        if self.cfg.train.resume_path:
            self._resume_training(self.cfg.train.resume_path)

    def _init_workspace(self):
        os.makedirs(self.cfg.project.output_dir, exist_ok=True)
        os.makedirs(self.cfg.project.logging_dir, exist_ok=True)

    def _init_wandb(self):
        id_file = os.path.join(self.cfg.project.output_dir, "wandb_id.txt")
        wandb_id = None
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                wandb_id = f.read().strip()

        run = wandb.init(
            project=self.cfg.project.name,
            dir=self.cfg.project.logging_dir,
            config=self.cfg,
            name=f"{self.cfg.project.name}-ep{self.cfg.train.epochs}",
            resume="allow",
            id=wandb_id
        )
        if not os.path.exists(id_file):
            with open(id_file, 'w') as f: f.write(run.id)

    def _setup_model_optimized(self):
        """加载模型并立即卸载 VAE 到 CPU"""
        logger.info(f"🚀 Loading WanModel from {self.cfg.model.path}...")
        dtype = torch.bfloat16
        
        self.model = WanModel.from_pretrained(
            self.cfg.model.path, 
            device=self.device, 
            dtype=dtype
        )
        
        # === 核心优化: VAE Offload ===
        # 既然数据已经是 Latent，训练时完全不需要 VAE
        logger.info("📉 Offloading VAE to CPU to save VRAM...")
        self.model.vae.to("cpu")
        self.model.vae.requires_grad_(False)
        
        # 确保 Transformer 在 GPU
        self.model.transformer.to(self.device)
        self.model.transformer.train()
        
        if self.cfg.model.use_lora:
            logger.info(f"💉 Injecting LoRA (Rank={self.cfg.model.lora_rank})...")
            lora_config = LoraConfig(
                r=self.cfg.model.lora_rank,
                lora_alpha=self.cfg.model.lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                bias="none"
            )
            self.model.transformer = get_peft_model(self.model.transformer, lora_config)
            self.model.transformer.print_trainable_parameters()
            self.params_to_optimize = filter(lambda p: p.requires_grad, self.model.transformer.parameters())
        else:
            logger.info("🔓 Unfreezing Transformer for Full-Parameter Training...")
            if self.cfg.train.gradient_checkpointing:
                self.model.transformer.enable_gradient_checkpointing()
            for param in self.model.transformer.parameters():
                param.requires_grad = True
            self.params_to_optimize = self.model.transformer.parameters()

    def _setup_optimizer(self):
        opt_type = self.cfg.train.optimizer
        if opt_type == "prodigy":
            logger.info(f"✨ Using Prodigy Optimizer (LR={self.cfg.train.lr})...")
            self.optimizer = prodigyopt.Prodigy(
                self.params_to_optimize,
                lr=self.cfg.train.lr,
                weight_decay=0.01,
                decouple=True,
                use_bias_correction=True,
                safeguard_warmup=True,
                d_coef=self.cfg.train.d_coef 
            )
        else:
            logger.info(f"Using AdamW Optimizer (LR={self.cfg.train.lr})...")
            self.optimizer = torch.optim.AdamW(self.params_to_optimize, lr=self.cfg.train.lr)

    def _setup_dataloader(self):
        # 1. 配置选项
        opt = Options()
        for k, v in self.cfg.dataset.items():
            if hasattr(opt, k): setattr(opt, k, v)
            
        # 2. 自动检查/生成缓存 (JIT Caching Logic)
        # 这会调用 dataset_wrapper 中的逻辑，如果没有缓存则生成，有则直接返回路径
        cache_dir = ensure_latents_cached(self.cfg, opt)
        
        # 3. 使用极速内存 Dataset (直接读取 .pt)
        self.dataset = InMemoryLatentDataset(cache_dir)
        
        logger.info(f"📚 Dataset Ready: {len(self.dataset)} samples in RAM.")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=0, # 内存读取设为 0 最快，避免 IPC 开销
            pin_memory=True
        )

    def compute_loss(self, z_0, z_1, prompts):
        """
        修正后的 compute_loss: 直接接受 Latents 不进行 Encode。
        Implementation strictly following Algorithm S1 in Supplementary Material.
        """
        s = self.cfg.inference.noise_scale
        
        # 0. Encode Prompts (文本编码依然需要)
        with torch.no_grad():
            prompt_embeds = self.model.encode_prompt(prompts)
            # [关键修改] 移除了 self.model.encode(pixels)
            # 此时 z_0, z_1 已经是 [B, C, T, H, W] 的 Normalized Latent

        B = z_0.shape[0]
        D = z_0[0].numel()
        
        # 1. Sample t & epsilon
        t = torch.rand((B,), device=self.device, dtype=z_0.dtype)
        epsilon = torch.randn_like(z_0)
        
        # 防止 t=0 或 t=1
        t_safe = torch.clamp(t.view(B, 1, 1, 1, 1), 1e-5, 1.0 - 1e-5)
        
        # 2. Construct Intermediate State x_t
        bridge_sigma = s * torch.sqrt(t_safe * (1 - t_safe))
        x_t = (1 - t_safe) * z_0 + t_safe * z_1 + bridge_sigma * epsilon
        
        # 3. Compute Velocity Target u_t
        u_t = (z_1 - x_t) / (1 - t_safe)
        
        # 4. Compute Normalization Factor Alpha
        diff_sq = torch.sum((z_1 - z_0) ** 2, dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-6)
        alpha = torch.sqrt(1 + (s**2 * t_safe * D) / ((1 - t_safe) * diff_sq))
        
        # 5. Model Forward (Time Flip: 0->1 => 1000->0)
        # 物理时间 t=0 (Source) 对应 模型时间 T=1000
        t_input = (1.0 - t) * 1000
        
        v_pred = self.model(
            hidden_states=x_t,
            timestep=t_input,
            encoder_hidden_states=prompt_embeds
        )
        
        # 6. Loss Calculation
        # Loss = || (v - u) / alpha ||^2
        loss = F.mse_loss(v_pred / alpha, u_t / alpha)
        return loss

    def train(self):
        self.model.transformer.train()
        accum_steps = self.cfg.train.gradient_accumulation_steps
        
        logger.info(f"🚀 Start Training: Epoch {self.start_epoch} -> {self.cfg.train.epochs}")
        
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            pbar = tqdm(self.dataloader, desc=f"Ep {epoch+1}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(pbar):
                # 直接获取 Latent (Dataset 返回的是字典)
                # 数据已经在 RAM 中，只需搬运到 GPU 并转为 bf16
                source = batch['source'].to(self.device, dtype=torch.bfloat16)
                target = batch['target'].to(self.device, dtype=torch.bfloat16)
                prompts = batch['prompt']
                
                # 计算 Loss (传入 Latent)
                loss = self.compute_loss(source, target, prompts)
                
                # 反向传播
                (loss / accum_steps).backward()
                current_loss = loss.item()
                epoch_loss += current_loss

                # EMA 平滑
                if self.avg_loss is None: self.avg_loss = current_loss
                else: self.avg_loss = 0.9 * self.avg_loss + 0.1 * current_loss
                
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.params_to_optimize, 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    if self.global_step % self.cfg.train.log_interval == 0:
                        wandb.log({
                            "train/loss": current_loss,
                            "train/loss_smooth": self.avg_loss,
                            "train/lr": self.optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch
                        })
                        pbar.set_postfix({"loss": f"{self.avg_loss:.4f}"})
                    
                    if self.global_step % self.cfg.train.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}", epoch)

            logger.info(f"🏁 Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.dataloader):.4f}")
            self.save_checkpoint(f"epoch_{epoch+1}", epoch)

    @torch.no_grad()
    def run_validation_sampling(self, tag):
        """
        验证逻辑需要解码视频，因此必须临时将 VAE 搬回 GPU。
        """
        if self.val_batch is None: return
        logger.info(f"🎨 Validation Sampling [{tag}]...")
        self.model.transformer.eval()
        
        # === 临时操作: VAE 回 GPU ===
        self.model.vae.to(self.device)
        
        try:
            # Dataset 返回的是 Latents
            source_latent = self.val_batch['source'][:1].to(self.device, dtype=torch.bfloat16)
            target_latent = self.val_batch['target'][:1].to(self.device, dtype=torch.bfloat16)
            prompt = self.val_batch['prompt'][0]
            
            # 手动执行推理 (使用 Backward Scheduler 1000->0)
            # 因为 generate_vibt 可能会试图对 source 进行 encode，而我们手里已经是 latent 了
            # 直接在这里写简单的 loop 更安全、可控
            
            from .scheduler import ViBTScheduler
            scheduler = ViBTScheduler(num_train_timesteps=1000)
            scheduler.set_parameters(
                noise_scale=self.cfg.inference.noise_scale, 
                shift_gamma=self.cfg.inference.shift_gamma, 
                seed=self.cfg.inference.seed
            )
            scheduler.set_timesteps(20, device=self.device)
            
            curr = source_latent.clone()
            prompt_emb = self.model.encode_prompt([prompt])
            
            for t in scheduler.timesteps:
                # 预测速度场
                t_model = t.unsqueeze(0).to(self.device)
                v = self.model(curr, t_model, prompt_emb)
                curr = scheduler.step(v, t, curr)[0]
            
            def decode_to_cpu(z):
                # 反归一化 Latent
                if hasattr(self.model.vae.config, "latents_mean"):
                    mean = torch.tensor(self.model.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
                    std = torch.tensor(self.model.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
                    z = z * std + mean
                
                # VAE Decode -> [1, 3, T, H, W]
                vid = self.model.vae.decode(z)[0]
                
                # Squeeze Batch -> [3, T, H, W]
                vid = vid.squeeze(0)
                
                # Pixel 反归一化 [-1, 1] -> [0, 255]
                vid = (vid * 0.5 + 0.5).clamp(0, 1) * 255
                return vid.to(torch.uint8).cpu() # 尽早转 CPU uint8 节省显存

            # 获取三个独立的 CPU Tensor [3, T, H, W]
            src_cpu = decode_to_cpu(source_latent)
            pred_cpu = decode_to_cpu(curr)
            tgt_cpu = decode_to_cpu(target_latent)
            
            # A. 本地保存 (拼接: Left|Mid|Right)
            # [3, T, H, 3*W]
            combined = torch.cat([src_cpu, pred_cpu, tgt_cpu], dim=3)
            
            # [T, H, W, C]
            local_vid = combined.permute(1, 2, 3, 0)
            
            local_dir = os.path.join(self.cfg.project.output_dir, "samples")
            os.makedirs(local_dir, exist_ok=True)
            save_path = os.path.join(local_dir, f"{tag}.mp4")
            
            try:
                torchvision.io.write_video(save_path, local_vid, fps=8)
                logger.info(f"💾 Saved combined sample to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save local video: {e}")

            # B. WandB 上传 (分屏展示)
            def to_wandb(tensor):
                return tensor.permute(1, 0, 2, 3).numpy()

            wandb.log({
                f"val/{tag}_source": wandb.Video(to_wandb(src_cpu), fps=8, format="mp4", caption="Source"),
                f"val/{tag}_pred":   wandb.Video(to_wandb(pred_cpu), fps=8, format="mp4", caption=f"Pred: {prompt}"),
                f"val/{tag}_target": wandb.Video(to_wandb(tgt_cpu), fps=8, format="mp4", caption="Target"),
                "global_step": self.global_step
            })
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            # === 恢复: VAE 回 CPU ===
            self.model.vae.to("cpu")
            self.model.transformer.train()
            torch.cuda.empty_cache()

    def save_checkpoint(self, tag, epoch):
        path = os.path.join(self.cfg.project.output_dir, f"checkpoint_{tag}")
        os.makedirs(path, exist_ok=True)
        
        self.model.transformer.save_pretrained(path)
        
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step
        }, os.path.join(path, "training_state.pt"))
        
        self.run_validation_sampling(tag)

    def _get_fixed_validation_batch(self):
        try: return next(iter(self.dataloader))
        except: return None

    def _resume_training(self, checkpoint_path):
        logger.info(f"🔄 Resuming from {checkpoint_path}...")
        if self.cfg.model.use_lora:
            from peft import set_peft_model_state_dict
            if os.path.exists(p := os.path.join(checkpoint_path, "adapter_model.bin")):
                set_peft_model_state_dict(self.model.transformer, torch.load(p, map_location=self.device))
        else:
            from safetensors.torch import load_file
            if os.path.exists(p := os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")):
                self.model.transformer.load_state_dict(load_file(p), strict=False)
        
        if os.path.exists(p := os.path.join(checkpoint_path, "training_state.pt")):
            state = torch.load(p, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_epoch = state["epoch"]
            self.global_step = state["global_step"]