import os
import sys
import math
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import prodigyopt
import logging
import torch._dynamo
import pytorch_lightning as L
from diffusers import (
    WanPipeline,
    DiffusionPipeline
)

DETAILED_FORMAT = (
    '%(asctime)s | '
    '%(levelname)-s | '
    '%(name)s | '
    '%(filename)s:%(lineno)d | '
    '%(funcName)s() | ' 
    # 'PID:%(process)d | TID:%(thread)d | '
    '%(message)s'
)
logging.basicConfig(
    level=logging.INFO,
    format=DETAILED_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

from .env import ViBTEnvConfig

class ViBTWanTrainer(L.LightningModule):
    def __init__(
        self,
        cfg: ViBTEnvConfig,
    ):
        # self.cfg = cfg
        self.name = cfg.project.name
        self.output_dir = cfg.project.output_dir
        self.logging_dir = cfg.project.logging_dir
        self.model_path = cfg.model.local_path
        self.use_lora = cfg.model.use_lora
        self.lora_rank = cfg.model.lora_rank
        self.lora_alpha = cfg.model.lora_alpha
        self.device = cfg.train.device
        self.dtype = cfg.train.dtype
        self.gradient_checkpointing = cfg.train.gradient_checkpointing
        self.avg_loss = None
        self.metrics_buffer = {}
        
        # 加载工作目录
        self._init_workspace()
        
        # 创建 wandb 监控
        self._init_wandb()
        
        # 加载模型
        self._setup_model()
        
    def _init_workspace(self):
        logger.info(f"🚀 Initializing project worksapce...")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        logger.info(f"✅ Initializing project worksapce OK!")
        
    def _init_wandb(self):
        logger.info(f"🚀 Initializing WanDB board...")
        id_file = os.path.join(self.output_dir, "wandb_id.txt")
        wandb_id = None
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                wandb_id = f.read().strip()

        run = wandb.init(
            project=self.name,
            dir=self.logging_dir,
            name=f"{self.name}",
            resume="allow",
            id=wandb_id
        )
        if not os.path.exists(id_file):
            with open(id_file, 'w') as f: f.write(run.id)
        logger.info(f"✅ Initializing WanDB board OK!")
    
    def _setup_model(self):
        logger.info(f"🚀 Loading Wan Pipeline from {self.model_path}...")
        self.pipe = WanPipeline.from_pretrained(
            self.model_path, 
            device=self.device, 
            dtype=self.dtype
        )
        logger.info(f"✅ Loading Wan Pipeline OK!")
        
        # VAE Offload
        logger.info("📉 Offloading VAE to CPU to save VRAM...")
        self.pipe.vae.to("cpu")
        self.pipe.vae.requires_grad_(False)
        logger.info(f"✅ Offloading VAE to CPU OK!")
        
        # set trandformer to be trained
        logger.info("✨ Move transformer to gpu and set train mode...")
        self.pipe.transformer.to(self.device)
        self.pipe.transformer.train()
        logger.info(f"✅ Moved and ready to be trained!")
        
        if self.use_lora:
            logger.info(f"💉 Injecting LoRA (Rank={self.lora_rank}, scale={self.lora_alpha})...")
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                bias="none"
            )
            self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
            self.pipe.transformer.print_trainable_parameters()
            self.params_to_optimize = list(filter(lambda p: p.requires_grad, self.pipe.transformer.parameters()))
            logger.info(f"✅ Injecting LoRA OK!")
        else:
            logger.info("🔓 Unfreezing Transformer for Full-Parameter Training...")
            if self.gradient_checkpointing:
                logger.info("✅ Enable gradient checkpoint: True")
                self.pipe.transformer.enable_gradient_checkpointing()
            for param in self.pipe.transformer.parameters():
                param.requires_grad = True
            self.params_to_optimize = list(self.model.transformer.parameters())
            logger.info("✅ Unfreezing Transformer OK!")
