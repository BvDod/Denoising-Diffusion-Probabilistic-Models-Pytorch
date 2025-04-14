from dataclasses import dataclass
from train import train_diffusion, DiffusionConfig


diff_config = DiffusionConfig(
    dataset_name = "FFHQ",
    learning_rate = 0.001,
    batch_size= 32,
    use_augmentation = False
)

train_diffusion(diff_config)

