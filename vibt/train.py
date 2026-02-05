import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torchvision
import prodigyopt
import logging

from .wan import WanModel
from .dataset_wrapper import Options
from .env import ViBTEnvConfig
from .inference import generate_vibt

DETAILED_FORMAT = (
    '%(asctime)s | '
    '%(levelname)-s | '
    '%(name)s | '
    '%(filename)s:%(lineno)d | '
    '%(funcName)s() | ' 
    'PID:%(process)d | '
    'TID:%(thread)d | '
    '%(message)s'
)
logging.basicConfig(
    level=logging.INFO,
    format=DETAILED_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

class ViBTTrainer:
    def __init__(
        self, 
        cfg: ViBTEnvConfig, 
        dataset: Dataset
    ):
        self.cfg = cfg
        self.device = "cuda"
        self.dataset = dataset
        self.avg_loss = None
        
        # 1. 环境初始化
        self._init_workspace()
        self._init_wandb()
        
        # 2. 模型加载
        self._setup_model()
        
        # 3. 优化器配置 (Prodigy / AdamW)
        self._setup_optimizer()
        
        # 4. 数据加载
        self._setup_dataloader()
        
        # 5. 状态恢复与验证准备
        self.val_batch = self._get_fixed_validation_batch()
        self.global_step = 0
        self.start_epoch = 0
        
        if self.cfg.train.resume_path:
            self._resume_training(self.cfg.train.resume_path)

    def _init_workspace(self):
        os.makedirs(self.cfg.project.output_dir, exist_ok=True)
        os.makedirs(self.cfg.project.logging_dir, exist_ok=True)

    def _init_wandb(self):
        """初始化 WandB，支持断点续传 ID"""
        id_file = os.path.join(self.cfg.project.output_dir, "wandb_id.txt")
        wandb_id = None
        
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                wandb_id = f.read().strip()

        run = wandb.init(
            project=self.cfg.project.name,
            dir=self.cfg.project.logging_dir,
            config=self.cfg, # OmegaConf object is serializable
            name=f"{self.cfg.project.name}-ep{self.cfg.train.epochs}",
            resume="allow",
            id=wandb_id
        )
        
        if not os.path.exists(id_file):
            with open(id_file, 'w') as f:
                f.write(run.id)

    def _setup_model(self):
        logger.info(f"🚀 Loading WanModel from {self.cfg.model.path}...")
        dtype = torch.bfloat16 if self.cfg.train.mixed_precision == "bf16" else torch.float16
        
        self.model = WanModel.from_pretrained(
            self.cfg.model.path, 
            device=self.device, 
            dtype=dtype
        )
        
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
        """配置优化器，优先支持 Prodigy"""
        opt_type = self.cfg.train.optimizer
        lr = self.cfg.train.lr
        d_coef = self.cfg.train.d_coef
        
        if opt_type == "prodigy":
            try:
                logger.info(f"✨ Using Prodigy Optimizer (LR={lr}, d_coef={d_coef})...")
                self.optimizer = prodigyopt.Prodigy(
                    self.params_to_optimize,
                    lr=lr,
                    weight_decay=0.01,
                    decouple=True,
                    use_bias_correction=True,
                    safeguard_warmup=True,
                    d_coef=d_coef 
                )
            except ImportError:
                logger.warning("⚠️ Prodigy not installed. Fallback to AdamW.")
                opt_type = "adamw"
        
        if opt_type == "adamw":
            logger.info(f"Using AdamW Optimizer (LR={lr})...")
            self.optimizer = torch.optim.AdamW(self.params_to_optimize, lr=lr)

    def _setup_dataloader(self):
        opt = Options()
        for k, v in self.cfg.dataset.items():
            if hasattr(opt, k): setattr(opt, k, v)
            
        logger.info(f"📚 Loading dataset: {len(self.dataset)} samples.")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def compute_loss(self, ego_pixel, exo_pixel, prompts):
        """
        Implementation strictly following Algorithm S1 in Supplementary Material.
        Ref: Tan et al., 2025, Vision Bridge Transformer at Scale, Page 16.
        """
        s = self.cfg.inference.noise_scale  # noise scale s 
        with torch.no_grad():
            prompt_embeds = self.model.encode_prompt(prompts)
            x_0 = self.model.encode(ego_pixel)  # Source x0
            x_1 = self.model.encode(exo_pixel)  # Target x1

        B = x_0.shape[0]
        D = x_0[0].numel() # latent dimension D 
        
        # "Sample ... interpolation time t ~ U(0,1), and noise epsilon ~ N(0,I)" [cite: 129, 577]
        t = torch.rand((B,), device=self.device, dtype=x_0.dtype)
        epsilon = torch.randn_like(x_0)
        
        # 为了数值稳定性，防止除以 0，这里做一个极小的 clamp
        # 虽然伪代码没写，但这是工程实现的必须步骤，否则 t=1 时 u_t 会爆炸
        t_expand = t.view(B, 1, 1, 1, 1)
        t_safe = torch.clamp(t_expand, min=1e-5, max=1.0 - 1e-5)
        
        # ==================== Step 3: Construct Intermediate State ====================
        # "Construct intermediate state x_t = (1-t)x_0 + t*x_1 + s*sqrt(t(1-t))*epsilon" 
        coeff_x0 = 1 - t_safe
        coeff_x1 = t_safe
        coeff_noise = s * torch.sqrt(t_safe * (1 - t_safe))
        
        x_t = coeff_x0 * x_0 + coeff_x1 * x_1 + coeff_noise * epsilon
        
        # ==================== Step 4: Compute Velocity Target ====================
        # "Compute velocity target u_t = (x_1 - x_t) / (1 - t)" [cite: 101, 132, 577]
        u_t = (x_1 - x_t) / (1 - t_safe)
        
        # ==================== Step 6: Compute Normalization Factor ====================
        # "Compute normalization factor alpha^2 = 1 + s^2*t*D / [(1-t)*||x_1 - x_0||^2]" 
        
        # ||x_1 - x_0||^2
        dist_sq = torch.sum((x_1 - x_0) ** 2, dim=[1, 2, 3, 4]).view(B, 1, 1, 1, 1)
        dist_sq = torch.clamp(dist_sq, min=1e-6) # 防止除零

        numerator = (s**2) * t_safe * D
        denominator = (1 - t_safe) * dist_sq
        alpha_sq = 1 + numerator / denominator
        alpha = torch.sqrt(alpha_sq)
        
        # ==================== Step 7: Compute Loss ====================
        # "L_velocity = || (v_theta(x_t, t) - u_t) / alpha ||^2" [cite: 136, 577]
        t_input = t * 1000
        v_pred = self.model(
            hidden_states=x_t,
            timestep=t_input,
            encoder_hidden_states=prompt_embeds
        )
        target_term = u_t / alpha
        pred_term = v_pred / alpha
        
        mse_loss = F.mse_loss(pred_term, target_term)
        loss = mse_loss * D
        
        return loss

    def train(self):
        self.model.transformer.train()
        accum_steps = self.cfg.train.gradient_accumulation_steps
        
        logger.info(f"🚀 Start Training: Epoch {self.start_epoch} -> {self.cfg.train.epochs}")
        
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            pbar = tqdm(self.dataloader, desc=f"Ep {epoch+1}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(pbar):
                source = batch['source_video'].to(self.device)
                target = batch['target_video'].to(self.device)
                prompts = [p for p in batch['prompt']] * source.shape[0]
                
                # 计算 Loss
                loss = self.compute_loss(source, target, prompts)
                
                # 反向传播
                (loss / accum_steps).backward()
                
                # 获取当前瞬时 Loss
                current_loss = loss.item()
                epoch_loss += current_loss

                # [新增] 计算 EMA 平滑 Loss
                if self.avg_loss is None:
                    self.avg_loss = current_loss
                else:
                    # 0.9 是平滑系数，值越大越平滑但滞后越高
                    self.avg_loss = 0.9 * self.avg_loss + 0.1 * current_loss
                
                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.params_to_optimize, 1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    if self.global_step % self.cfg.train.log_interval == 0:
                        wandb.log({
                            "train/loss": current_loss,           # 原始数据
                            "train/loss_smooth": self.avg_loss,   # 平滑曲线
                            "train/lr": self.optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch
                        })
                        # 在进度条中显示平滑 Loss，看起来更稳定
                        pbar.set_postfix({"loss": f"{self.avg_loss:.4f}"})
                    
                    if self.global_step % self.cfg.train.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}", epoch)

            logger.info(f"🏁 Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.dataloader):.4f}")
            self.save_checkpoint(f"epoch_{epoch+1}", epoch)

    @torch.no_grad()
    def run_validation_sampling(self, tag):
        """执行验证采样，生成可视化结果"""
        if self.val_batch is None: return
        
        logger.info(f"🎨 Validation Sampling [{tag}]...")
        self.model.transformer.eval()
        
        try:
            # 准备单个样本
            source = self.val_batch['source_video'][:1] # [1, C, F, H, W]
            target_gt = self.val_batch['target_video'][:1].to(self.device)
            prompt = self.val_batch['prompt'][0]
            
            # 调用 inference.py 中的生成逻辑 (复用 Scheduler 和 Pipeline)
            # 注意：generate_vibt 内部会处理 Normalization 和 Scheduler
            pred_tensor = generate_vibt(
                model=self.model,
                source_input=source, # 传入 Tensor ([-1, 1])
                prompt=prompt,
                steps=20, # 快速验证
                device=self.device,
                shift_gamma=self.cfg.inference.shift_gamma,
                noise_scale=self.cfg.inference.noise_scale,
                guidance_scale=1.5
            )
            
            pred_tensor = pred_tensor.to(self.device)
            
            # 拼接视频: Source | Pred | GT
            # 维度调整 [B, C, F, H, W] -> WandB 格式
            def _prep_vid(v):
                v = v[0].permute(1, 0, 2, 3) # [F, C, H, W]
                v = (v * 0.5 + 0.5).clamp(0, 1) * 255
                return v.to(torch.uint8).cpu().numpy()

            wandb.log({
                "val/comparison": wandb.Video(
                    _prep_vid(torch.cat([source.to(self.device), pred_tensor, target_gt], dim=4)), 
                    fps=8, format="mp4", caption=f"{tag}: {prompt}"
                ),
                "global_step": self.global_step
            })
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.model.transformer.train()

    def save_checkpoint(self, tag, epoch):
        path = os.path.join(self.cfg.project.output_dir, f"checkpoint_{tag}")
        os.makedirs(path, exist_ok=True)
        logger.info(f"💾 Saving to {path}...")
        
        # 保存模型权重
        self.model.transformer.save_pretrained(path)
        
        # 保存训练状态
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step
        }, os.path.join(path, "training_state.pt"))
        
        # 触发验证
        self.run_validation_sampling(tag)

    def _get_fixed_validation_batch(self):
        try:
            return next(iter(self.dataloader))
        except:
            return None

    def _resume_training(self, checkpoint_path):
        logger.info(f"🔄 Resuming from {checkpoint_path}...")
        
        # 1. 权重
        # 优先尝试 safetensors, 其次 bin
        if self.cfg.model.use_lora:
            from peft import set_peft_model_state_dict
            if os.path.exists(p := os.path.join(checkpoint_path, "adapter_model.bin")):
                set_peft_model_state_dict(self.model.transformer, torch.load(p, map_location=self.device))
        else:
            from safetensors.torch import load_file
            if os.path.exists(p := os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")):
                self.model.transformer.load_state_dict(load_file(p), strict=False)
        
        # 2. 状态
        if os.path.exists(p := os.path.join(checkpoint_path, "training_state.pt")):
            state = torch.load(p, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.start_epoch = state["epoch"]
            self.global_step = state["global_step"]
            logger.info(f"   Resumed at Epoch {self.start_epoch}, Step {self.global_step}")