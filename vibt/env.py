import os
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

# 加载环境变量
load_dotenv()
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# -----------------------------------------------------------------------------
# 0. 路径定义
# -----------------------------------------------------------------------------
_CONFIGS_DIR = Path(__file__).resolve().parent      # .../configs
_PACKAGE_DIR = _CONFIGS_DIR.parent                  # .../package_root
_PROJECT_ROOT = _PACKAGE_DIR.parent                 # .../ (Repo Root)

# 导出给其他模块使用
globals()["PROJECT_ROOT"] = str(_PROJECT_ROOT)
globals()["PACKAGE_DIR"] = str(_PACKAGE_DIR)
globals()["CONFIG_DIR"] = str(_CONFIGS_DIR)

# -----------------------------------------------------------------------------
# 1. 定义配置 Schema
# -----------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    name: str = "ViBT-Ego2Exo"
    output_dir: str = "outputs/"
    logging_dir: str = "logs/"

@dataclass
class ModelConfig:
    path: str = "/opt/liblibai-models/user-workspace2/model_zoo/Wan2.1-T2V-1.3B"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32

@dataclass
class DatasetConfig:
    repo_id: str = "Kytolly/examples_Ego2ExoFollowCamera"
    root: str = "/opt/liblibai-models/user-workspace2/dataset/ego/FollowBench"
    index: str = "index.json"
    phase: str = "train"
    clip_len: int = 600
    stride: int = 4
    height: int = 704
    width: int = 1280
    batch_size: int = 1
    serial_batches: bool = True
    num_workers: int = 4

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    epochs: int = 20
    gradient_accumulation_steps: int = 4
    save_interval: int = 100
    log_interval: int = 10
    mixed_precision: str = "bf16"
    resume_path: str = ""
    gradient_checkpointing: bool = True
    instruction: str = "Transform the view from ego-centric to third-person perspective"

@dataclass
class ViBTEnvConfig:
    """环境配置基类"""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# -----------------------------------------------------------------------------
# 2. 配置加载逻辑
# -----------------------------------------------------------------------------

def load_config(config_file_name: str = "config/video2video.yaml"):
    """
    加载配置并将相对路径解析为相对于 PROJECT_ROOT 的绝对路径。
    """
    
    # --- A. 寻找配置文件 ---
    config_path = os.getenv("CONFIG_PATH", config_file_name)
    path_to_use = None
    p = Path(config_path)

    if p.is_absolute():
        if p.exists(): path_to_use = p
    else:
        candidates = [
            _PROJECT_ROOT / config_path,
            _CONFIGS_DIR / config_path,
            Path.cwd() / config_path
        ]
        
        for candidate in candidates:
            if candidate.exists():
                path_to_use = candidate
                break

    if path_to_use is None:
        path_to_use = _PROJECT_ROOT / config_file_name
        if not path_to_use.exists():
             print(f"Warning: Config file '{config_path}' not found. Using default Schema values.")

    # --- B. 加载 YAML 并 Merge Schema ---
    if path_to_use and path_to_use.exists():
        yaml_cfg = OmegaConf.load(path_to_use)
    else:
        yaml_cfg = OmegaConf.create({})

    schema_cfg = OmegaConf.structured(ViBTEnvConfig)
    cfg = OmegaConf.merge(schema_cfg, yaml_cfg)

    # --- C. 路径规范化逻辑 (相对于 PROJECT_ROOT) ---
    def _is_path_like(key: str, val: str):
        """启发式判断是否是路径字段"""
        if not isinstance(val, str): return False
        if Path(val).is_absolute(): return False 
        
        key = (key or "").lower()
        
        # [核心修复] 移除了 "index"，防止 index.json 被错误解析为项目根目录文件
        # 保留了 path, dir, file, root, output, log
        keywords = ["path", "dir", "file", "root", "output", "log"]
        
        if any(k in key for k in keywords): return True
        if ("/" in val) or ("\\" in val) or val.startswith("."): return True
        return False

    def _normalize(obj: object, parent_key: str = "") -> object:
        if isinstance(obj, dict):
            return {k: _normalize(v, k) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)([_normalize(x, parent_key) for x in obj])
        
        if isinstance(obj, str) and _is_path_like(parent_key, obj):
            try:
                abs_path = (_PROJECT_ROOT / obj).resolve()
                return str(abs_path)
            except Exception:
                return str(_PROJECT_ROOT / obj)
                
        return obj

    container = OmegaConf.to_container(cfg, resolve=True)
    normalized_container = _normalize(container)
    final_cfg = OmegaConf.create(normalized_container)
    
    OmegaConf.set_readonly(final_cfg, True)
    globals()["CONFIG_FILE"] = str(path_to_use)

    return final_cfg

CONFIG: DictConfig = load_config()