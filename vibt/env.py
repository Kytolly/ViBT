from dataclasses import dataclass, field
import torch

@dataclass
class ProjectConfig:
    name: str = "ViBT-Project"
    output_dir: str = "outputs/"
    logging_dir: str = "logs/"

@dataclass
class ModelConfig:
    local_path: str = "/opt/liblibai-models/user-workspace2/model_zoo/Wan2.1-T2V-1.3B"
    use_lora: bool = True
    lora_rank: int = 128
    lora_alpha: int = 128
    
@dataclass
class TrainConfig:
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    gradient_checkpointing: bool = False
    
@dataclass
class ViBTEnvConfig:
    """环境配置基类"""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)