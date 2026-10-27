"""Dependency-free configuration contracts; defaults are design starting points."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldModelConfig:
    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    window: int = 8
    batches: int = 10
    batch_size: int = 2048
    learning_rate: float = 2e-4


@dataclass(frozen=True)
class ImaginationConfig:
    horizon: int = 32
    batch_size: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2e-4


@dataclass(frozen=True)
class CollectionConfig:
    num_envs: int = 1024
    buffer_steps: int = 1000
    update_to_data_ratio: float = 0.4
    action_noise_std: float = 0.1
    world_model_only_iterations: int = 500


@dataclass(frozen=True)
class ExperimentConfig:
    task: str = "amp_flat"
    seed: int = 42
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
