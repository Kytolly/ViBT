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
# 导入推理模块用于验证采样
try:
    from vibt.inference import generate_vibt
except ImportError:
    generate_vibt = None # 容错处理

class ViBTTrainer:
    def __init__(self, cfg):
        """
        Args:
            cfg (DictConfig): 由 vibt.env.CONFIG 提供的全局配置对象
        """
        self.cfg = cfg
        self.device = "cuda"
        
        # [核心修复] 1. 先创建输出目录，防止 _init_wandb 写入 wandb_id.txt 时报错
        logger.info(f"📂 Creating directories: {self.cfg.project.output_dir}")
        os.makedirs(self.cfg.project.output_dir, exist_ok=True)
        os.makedirs(self.cfg.project.logging_dir, exist_ok=True)
        
        # 2. 初始化 WandB (现在目录已存在，可以安全写入)
        self._init_wandb()
        
        # 3. 加载模型
        logger.info(f"🚀 Loading WanModel from {self.cfg.model.path}...")
        dtype = torch.bfloat16 if self.cfg.training.mixed_precision == "bf16" else torch.float16
        
        self.model = WanModel.from_pretrained(
            self.cfg.model.path, 
            device=self.device,
            dtype=dtype
        )
        
        # 4. 配置训练模式
        if self.cfg.model.use_lora:
            self._setup_lora()
        else:
            # 解冻 Transformer 进行全量训练
            if self.cfg.training.gradient_checkpointing:
                self.model.transformer.enable_gradient_checkpointing()
            logger.info("🔓 Unfreezing Transformer for Full-Parameter Training...")
            for param in self.model.transformer.parameters():
                param.requires_grad = True
            
        # 5. 准备优化器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.transformer.parameters()),
            lr=self.cfg.training.lr
        )
        
        # 6. 准备数据
        self._setup_dataloader()
        
        # 7. 准备固定验证样本
        self.val_batch = self._get_fixed_validation_batch()
        
        self.global_step = 0
        self.start_epoch = 0

        # 8. 尝试恢复训练
        resume_path = getattr(self.cfg.training, "resume_path", "")
        if resume_path:
            self._resume_training(resume_path)

    def _init_wandb(self):
        """初始化 WandB"""
        resume_path = getattr(self.cfg.training, "resume_path", "")
        resume_mode = "allow" if resume_path else None
        
        id_file = os.path.join(self.cfg.project.output_dir, "wandb_id.txt")
        wandb_id = None
        
        if resume_path and os.path.exists(id_file):
            with open(id_file, 'r') as f:
                wandb_id = f.read().strip()

        # 初始化 run
        run = wandb.init(
            project=self.cfg.project.name,
            dir=self.cfg.project.logging_dir,
            config=self._flatten_config(self.cfg),
            name=f"vibt-lora-ep{self.cfg.training.epochs}",
            resume=resume_mode,
            id=wandb_id
        )
        
        # 保存 ID 供后续恢复
        if not os.path.exists(id_file):
            with open(id_file, 'w') as f:
                f.write(run.id)

    def _flatten_config(self, cfg):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)

    def _setup_lora(self):
        """注入 LoRA 模块"""
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
        """初始化数据集"""
        opt = Options()
        opt.assets = self.cfg.dataset.root
        opt.phase = self.cfg.dataset.phase
        opt.annotation = self.cfg.dataset.index
        opt.index = self.cfg.dataset.index
        
        opt.height = self.cfg.dataset.height
        opt.width = self.cfg.dataset.width
        opt.clip_len = self.cfg.dataset.clip_len
        
        logger.info(f"📚 Loading dataset from {opt.assets} ({opt.phase})...")
        dataset = FollowBenchDatasetWrapper(opt)
        logger.info(f"✅ Dataset loaded: {len(dataset)} samples.")
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True
        )

    def _get_fixed_validation_batch(self):
        """获取一个固定的 Batch 用于可视化"""
        logger.info("🖼️ Fetching a fixed validation batch...")
        try:
            batch = next(iter(self.dataloader))
            return batch
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch validation batch: {e}")
            return None

    def _resume_training(self, checkpoint_path):
        """恢复训练逻辑"""
        logger.info(f"🔄 Resuming training from {checkpoint_path}...")
        
        # 加载权重
        adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
        if os.path.exists(adapter_path):
            adapters_weights = torch.load(adapter_path, map_location=self.device)
            set_peft_model_state_dict(self.model.transformer, adapters_weights)
            logger.info("   ✅ Loaded LoRA weights.")
        else:
            from safetensors.torch import load_file
            adapter_path_safe = os.path.join(checkpoint_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path_safe):
                adapters_weights = load_file(adapter_path_safe)
                set_peft_model_state_dict(self.model.transformer, adapters_weights)
                logger.info("   ✅ Loaded LoRA weights (safetensors).")
            else:
                logger.warning(f"   ⚠️ weights not found, starting random init.")

        # 加载优化器
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info("   ✅ Loaded optimizer state.")

        # 加载进度
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.start_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            logger.info(f"   ✅ Resumed from Epoch {self.start_epoch}, Step {self.global_step}.")

    @torch.no_grad()
    def run_validation_sampling(self, step_tag):
        """运行验证采样并记录到 WandB"""
        if self.val_batch is None:
            return

        logger.info(f"🎨 Running validation sampling for {step_tag}...")
        self.model.transformer.eval() 

        # 1. 准备数据
        ego_video = self.val_batch['ego_video'][:1].to(self.device) 
        exo_video_gt = self.val_batch['exo_video'][:1].to(self.device)
        
        # 2. 采样推理 (简化版 Euler)
        # 这里直接复现 inference.py 的逻辑，避免文件读取问题
        prompt = self.cfg.training.instruction
        logger.info(f"   Using prompt: {prompt}")
        prompt_embeds = self.model.encode_prompt([prompt])
        
        # Encode
        z_curr = self.model.encode(ego_video)
        
        # Euler Loop
        steps = 20
        dt = 1.0 / steps
        for i in range(steps):
            t_curr = i / steps
            t_input = torch.tensor([t_curr * 1000], device=self.device, dtype=z_curr.dtype)
            pred_v = self.model(z_curr, t_input, prompt_embeds)
            z_curr = z_curr + pred_v * dt
            
        # Decode
        pred_video = self.model.decode(z_curr) # [1, C, F, H, W] in [-1, 1]
        
        # 处理为 WandB 格式 [0, 255]
        def process_for_wandb(vid_tensor):
            vid = vid_tensor[0].permute(1, 0, 2, 3) # [F, C, H, W]
            vid = (vid * 0.5 + 0.5).clamp(0, 1)     # [-1, 1] -> [0, 1]
            vid = (vid * 255).to(torch.uint8)
            return vid.cpu().numpy()

        wandb.log({
            "val/source_ego": wandb.Video(process_for_wandb(ego_video), fps=8, format="mp4"),
            "val/generated_exo": wandb.Video(process_for_wandb(pred_video), fps=8, format="mp4"),
            "val/ground_truth": wandb.Video(process_for_wandb(exo_video_gt), fps=8, format="mp4"),
            "global_step": self.global_step
        })
        
        self.model.transformer.train() 
        logger.info("   ✅ Validation sampling done.")

    def save_checkpoint(self, tag, epoch=None):
        """保存检查点"""
        path = os.path.join(self.cfg.project.output_dir, f"checkpoint_{tag}")
        os.makedirs(path, exist_ok=True)
        
        # 1. 保存模型
        self.model.transformer.save_pretrained(path)
        
        # 2. 保存优化器
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        
        # 3. 保存状态
        current_epoch = epoch if epoch is not None else self.start_epoch
        torch.save({
            "epoch": current_epoch,
            "global_step": self.global_step
        }, os.path.join(path, "training_state.pt"))
        
        logger.info(f"💾 Checkpoint saved: {path}")
        
        # 4. 验证采样
        try:
            self.run_validation_sampling(tag)
        except Exception as e:
            logger.warning(f"⚠️ Validation sampling failed: {e}")

    def compute_loss(self, ego_pixel, exo_pixel, prompts):
        """Bridge Matching Loss"""
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
        """训练主循环"""
        self.model.transformer.train()
        total_epochs = self.cfg.training.epochs
        accum_steps = self.cfg.training.gradient_accumulation_steps
        
        logger.info(f"🚀 Start Training from Epoch {self.start_epoch}...")
        
        for epoch in range(self.start_epoch, total_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(pbar):
                ego_video = batch['ego_video'].to(self.device)
                exo_video = batch['exo_video'].to(self.device)
                
                current_prompt = self.cfg.training.instruction
                prompts = [current_prompt] * ego_video.shape[0]
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
                
                current_loss = loss.item() * accum_steps
                epoch_loss += current_loss
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                
                if self.global_step > 0 and self.global_step % self.cfg.training.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}", epoch=epoch)

            self.save_checkpoint(f"epoch_{epoch+1}", epoch=epoch+1)
            logger.info(f"🏁 Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.dataloader):.4f}")