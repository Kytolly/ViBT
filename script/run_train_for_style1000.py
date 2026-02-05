import os
import sys
import logging
import torch
from omegaconf import OmegaConf

# ====================================================
# 1. 环境与路径设置
# ====================================================
# 确保当前目录在 python path 中，以便导入 vibt 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入项目模块
from vibt.env import ViBTEnvConfig
from vibt.train import ViBTTrainer

def setup_logging():
    """配置全局日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

def main():
    setup_logging()
    logger = logging.getLogger("Launcher")

    # ====================================================
    # 2. 加载与验证配置
    # ====================================================
    config_path = os.path.join(project_root, "config/stylization.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"❌ Configuration file not found at: {config_path}")
        sys.exit(1)

    logger.info(f"📜 Loading configuration from {config_path}...")
    
    # 加载 YAML
    file_conf = OmegaConf.load(config_path)
    
    # 加载默认 Schema (ViBTEnvConfig)
    schema = OmegaConf.structured(ViBTEnvConfig)
    
    # 合并: Schema(默认值) <- YAML(用户值)
    # 这步操作会自动进行类型检查，如果 YAML 里有未知字段会报错，防止配错参数
    try:
        cfg = OmegaConf.merge(schema, file_conf)
    except Exception as e:
        logger.error(f"❌ Configuration Error: {e}")
        sys.exit(1)

    # ====================================================
    # 3. 特定任务覆盖 (Optional Overrides)
    # ====================================================
    # 如果你是专门为 Style1000 跑的，可以在这里硬编码一些强制参数
    # 或者留空，完全由 yaml 控制。
    # 例如：强制开启 Prodigy
    # if cfg.train.optimizer != "prodigy":
    #     logger.warning("Force enabling Prodigy optimizer for Style1000 task!")
    #     cfg.train.optimizer = "prodigy"

    # 打印关键信息确认
    logger.info("-" * 40)
    logger.info(f"🔫 Task:       {cfg.project.name}")
    logger.info(f"📦 Model:      {cfg.model.path}")
    logger.info(f"📉 Optimizer:  {cfg.train.optimizer} (LR={cfg.train.lr})")
    logger.info(f"💾 Output Dir: {cfg.project.output_dir}")
    logger.info("-" * 40)

    # ====================================================
    # 4. 启动训练器
    # ====================================================
    # 设置随机种子 (如果配置中有)
    if hasattr(cfg.train, 'seed') and cfg.train.seed is not None:
        torch.manual_seed(cfg.train.seed)
        
    try:
        logger.info("🛠️  Initializing ViBTTrainer (Auto JIT Caching Enabled)...")
        # Trainer 内部会自动触发 ensure_latents_cached
        trainer = ViBTTrainer(cfg)
        
        logger.info("🚀 Starting Training Loop...")
        trainer.train()
        
        logger.info("✅ Training Finished Successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\n🛑 Training interrupted by user (Ctrl+C).")
        # 可以在这里添加保存 checkpoint 的逻辑，但 Trainer 通常会定期保存
        sys.exit(0)
        
    except Exception as e:
        logger.exception(f"❌ Critical Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()