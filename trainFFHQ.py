from dataclasses import dataclass
from train import train_diffusion, DiffusionConfig, ModelConfig


diff_config = DiffusionConfig(
    dataset_name = "FFHQ",
    learning_rate = 0.001,
    batch_size= 32,
    use_augmentation = False,
    timesteps_diff = 200,
)

model_config = ModelConfig()

train_diffusion(diff_config, model_config)

