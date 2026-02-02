import os
import sys

# -----------------------------------------------------------------------------
# 1. 路径 Hack: 确保能导入项目根目录下的模块
# -----------------------------------------------------------------------------
# scripts/run_train.py -> scripts/ -> PROJECT_ROOT
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# -----------------------------------------------------------------------------
# 2. 导入配置和训练器
# -----------------------------------------------------------------------------
try:
    # 导入 vibt.env 时，它会自动读取 CONFIG_PATH 环境变量或默认 YAML
    # 并生成全局单例 CONFIG 对象
    from vibt.env import CONFIG
    from vibt.train import ViBTTrainer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"   Current sys.path: {sys.path}")
    sys.exit(1)

def main():
    print(f"==================================================")
    print(f"   ViBT Training Launcher")
    print(f"==================================================")
    print(f"📂 Project Root: {project_root}")
    print(f"📄 Config File:  {os.getenv('CONFIG_PATH', 'Default')}")
    print(f"🔧 Experiment:   {CONFIG.project.name}")
    print(f"💾 Output Dir:   {CONFIG.project.output_dir}")
    print(f"📊 Logging Dir:  {CONFIG.project.logging_dir}")
    print(f"==================================================\n")

    # 初始化训练器 (传入全局配置)
    trainer = ViBTTrainer(CONFIG)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()