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

from vibt.env import CONFIG
from vibt.train import ViBTTrainer
from vibt.dataset_wrapper import Options, FollowBenchDatasetWrapper

def main():
    print(f"==================================================")
    print(f"   ViBT Training Launcher")
    print(f"==================================================")
    print(f"📂 Project Root: {CONFIG.project.root}")
    print(f"📄 Config File:  {os.getenv('CONFIG_PATH', 'Default')}")
    print(f"🔧 Experiment:   {CONFIG.project.name}")
    print(f"💾 Output Dir:   {CONFIG.project.output_dir}")
    print(f"📊 Logging Dir:  {CONFIG.project.logging_dir}")
    print(f"==================================================\n")

    # 初始化训练器 (传入全局配置)
    opt = Options()
    opt.root = CONFIG.dataset.root
    opt.phase = CONFIG.dataset.phase
    opt.index = CONFIG.dataset.index
    opt.height = CONFIG.dataset.height
    opt.width = CONFIG.dataset.width
    opt.clip_len = CONFIG.dataset.clip_len
    opt.stride = CONFIG.dataset.stride
    
    dataset = FollowBenchDatasetWrapper(opt)
    trainer = ViBTTrainer(CONFIG, dataset)
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()