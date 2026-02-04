import os
import sys
import logging
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
from vibt.dataset_wrapper import Options, Style1000DatasetWrapper

# -----------------------------------------------------------------------------
# 2. 导入配置和训练器
# -----------------------------------------------------------------------------
try:
    from vibt.env import CONFIG_STYLIZATION
    from vibt.train import ViBTTrainer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"   Current sys.path: {sys.path}")
    sys.exit(1)

def main():
    print(f"==================================================")
    print(f"   ViBT Training Launcher")
    print(f"==================================================")
    print(f"📂 Project Root: {CONFIG_STYLIZATION.project.root}")
    print(f"📄 Config File:  {os.getenv('CONFIG_PATH', 'Default')}")
    print(f"🔧 Experiment:   {CONFIG_STYLIZATION.project.name}")
    print(f"💾 Output Dir:   {CONFIG_STYLIZATION.project.output_dir}")
    print(f"📊 Logging Dir:  {CONFIG_STYLIZATION.project.logging_dir}")
    print(f"==================================================\n")

    # 初始化训练器 (传入全局配置)
    opt = Options()
    opt.root = CONFIG_STYLIZATION.dataset.root
    opt.phase = CONFIG_STYLIZATION.dataset.phase
    opt.index = CONFIG_STYLIZATION.dataset.index
    opt.height = CONFIG_STYLIZATION.dataset.height
    opt.width = CONFIG_STYLIZATION.dataset.width
    opt.clip_len = CONFIG_STYLIZATION.dataset.clip_len
    opt.stride = CONFIG_STYLIZATION.dataset.stride
    
    dataset = Style1000DatasetWrapper(opt)
    trainer = ViBTTrainer(CONFIG_STYLIZATION, dataset)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()