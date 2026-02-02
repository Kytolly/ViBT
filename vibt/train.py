import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

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
        
        # 1. 初始化 WandB
        self._init_wandb()
        
        # 2. 创建输出目录
        os.makedirs(self.cfg.project.output_dir, exist_ok=True)
        
        # 3. 加载模型
        print(f"🚀 Loading WanModel from {self.cfg.model.path}...")
        # 根据配置决定精度
        dtype = torch.bfloat16 if self.cfg.training.mixed_precision == "bf16" else torch.float16
        
        self.model = WanModel.from_pretrained(
            self.cfg.model.path, 
            device=self.device,
            dtype=dtype
        )
        
        # 4. 配置 LoRA
        if self.cfg.model.use_lora:
            self._setup_lora()
            
        # 5. 准备优化器
        # 仅优化 Transformer 中 requires_grad 的参数 (即 LoRA 参数)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.transformer.parameters()),
            lr=self.cfg.training.lr
        )
        
        # 6. 准备数据
        self._setup_dataloader()
        
        self.global_step = 0
        self.start_epoch = 0

        # 7. 尝试恢复训练
        # 注意：这里假设您的 env.py Schema 中包含 training.resume_path
        # 如果没有，请确保在 env.py 中添加该字段，或使用 .get() 方法
        resume_path = getattr(self.cfg.training, "resume_path", "")
        if resume_path:
            self._resume_training(resume_path)

    def _init_wandb(self):
        """初始化 WandB"""
        # 检查是否是恢复训练
        resume_path = getattr(self.cfg.training, "resume_path", "")
        resume_mode = "allow" if resume_path else None
        
        id_file = os.path.join(self.cfg.project.output_dir, "wandb_id.txt")
        wandb_id = None
        
        # 尝试读取旧 ID 以保持曲线连续
        if resume_path and os.path.exists(id_file):
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
        
        # 保存 ID 供后续恢复使用
        if not os.path.exists(id_file):
            with open(id_file, 'w') as f:
                f.write(run.id)

    def _flatten_config(self, cfg):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)

    def _setup_lora(self):
        """注入 LoRA 模块"""
        print(f"💉 Injecting LoRA adapters (Rank={self.cfg.model.lora_rank})...")
        lora_config = LoraConfig(
            r=self.cfg.model.lora_rank,
            lora_alpha=self.cfg.model.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"], # 针对 Wan Transformer Attention
            bias="none"
        )
        self.model.transformer = get_peft_model(self.model.transformer, lora_config)
        self.model.transformer.print_trainable_parameters()

    def _setup_dataloader(self):
        """初始化数据集"""
        # 构造 DatasetWrapper 需要的 Options 对象
        opt = Options()
        
        # [核心] 将 CONFIG 映射到 Options
        opt.assets = self.cfg.dataset.root       # 数据集根目录
        opt.phase = self.cfg.dataset.phase       # train/test
        opt.annotation = self.cfg.dataset.index  # index.json
        opt.index = self.cfg.dataset.index       # 兼容性字段
        
        opt.height = self.cfg.dataset.height
        opt.width = self.cfg.dataset.width
        opt.clip_len = self.cfg.dataset.clip_len
        
        print(f"📚 Loading dataset from {opt.assets} ({opt.phase})...")
        dataset = FollowBenchDatasetWrapper(opt)
        print(f"✅ Dataset loaded: {len(dataset)} samples.")
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True
        )

    def _resume_training(self, checkpoint_path):
        """恢复训练逻辑"""
        print(f"🔄 Resuming training from {checkpoint_path}...")
        
        # 1. 加载 LoRA 权重
        adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
        if os.path.exists(adapter_path):
            adapters_weights = torch.load(adapter_path, map_location=self.device)
            set_peft_model_state_dict(self.model.transformer, adapters_weights)
            print("   ✅ Loaded LoRA weights.")
        else:
            # 尝试 safetensors
            from safetensors.torch import load_file
            adapter_path_safe = os.path.join(checkpoint_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path_safe):
                adapters_weights = load_file(adapter_path_safe)
                set_peft_model_state_dict(self.model.transformer, adapters_weights)
                print("   ✅ Loaded LoRA weights (safetensors).")
            else:
                print(f"   ⚠️ LoRA weights not found, starting random init.")

        # 2. 加载优化器
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            print("   ✅ Loaded optimizer state.")

        # 3. 加载进度
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.start_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            print(f"   ✅ Resumed from Epoch {self.start_epoch}, Step {self.global_step}.")

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
        
        print(f"💾 Checkpoint saved: {path}")

    def compute_loss(self, ego_pixel, exo_pixel, prompts):
        """Bridge Matching Loss"""
        # A. 编码 Text
        prompt_embeds = self.model.encode_prompt(prompts)
        
        # B. 编码 Video (VAE)
        with torch.no_grad():
            z_0 = self.model.encode(ego_pixel) 
            z_1 = self.model.encode(exo_pixel) 
            
        # C. Bridge Matching
        B = z_0.shape[0]
        t = torch.rand((B,), device=self.device, dtype=z_0.dtype)
        t_expand = t.view(B, 1, 1, 1, 1)
        
        z_t = (1 - t_expand) * z_0 + t_expand * z_1
        target_v = z_1 - z_0
        
        t_input = t * 1000 # Wan 时间步缩放
        
        # D. 预测
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
        
        print(f"🚀 Start Training from Epoch {self.start_epoch}...")
        
        for epoch in range(self.start_epoch, total_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(pbar):
                ego_video = batch['ego_video'].to(self.device)
                exo_video = batch['exo_video'].to(self.device)
                
                # 构造 Prompt (实际应从 batch['prompt'] 获取)
                prompts = ["Transform ego view to third-person view"] * ego_video.shape[0]

                # Forward
                loss = self.compute_loss(ego_video, exo_video, prompts)
                
                # Backward
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
                
                # 定期保存
                if self.global_step > 0 and self.global_step % self.cfg.training.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}", epoch=epoch)

            # Epoch 结束保存
            self.save_checkpoint(f"epoch_{epoch+1}", epoch=epoch+1)
            print(f"🏁 Epoch {epoch+1} Avg Loss: {epoch_loss / len(self.dataloader):.4f}")