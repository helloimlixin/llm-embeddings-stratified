from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class WandbConfig:
    project: str = "moe-sentiment"
    entity: str = None  # Set to your wandb username/team
    log_model: bool = True

@dataclass
class DataConfig:
    samples_per_domain: int = 50
    batch_size: int = 16
    pca_dim: int = 64
    embedding_model: str = "text-embedding-ada-002"

@dataclass
class ModelConfig:
    input_dim: int = 64
    hidden_dim: int = 128
    code_dim: int = 32
    num_experts: int = 3
    learning_rate: float = 1e-3
    lambda_reg: float = 1e-5

@dataclass
class TrainerConfig:
    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int = 1
    default_root_dir: str = "experiments"

@dataclass
class Config:
    wandb: WandbConfig = WandbConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    experiment_name: str = "moe_default"

# Register configs
cs = ConfigStore.instance()
cs.store(name="config", node=Config)