from train import train_diffusion, DiffusionConfig, ModelConfig
from sample import sample_images

diff_config = DiffusionConfig(
    dataset_name = "FFHQ",
    learning_rate = 0.00003,
    batch_size= 20,
    use_augmentation = False,
    timesteps_diff = 200,
)
model_config = ModelConfig(
    lowest_resolution_size = 8,
    dim_multiply = [1,2,2,4,8],
    transformer_layers = [3],
    base_dim = 64
)
sample_images(diff_config, model_config)

