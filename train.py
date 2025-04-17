from dataclasses import dataclass
from functions.dataHandling import get_dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functions.visualize import plot_grid_samples_tensor
import torch
from functions.noise_scheduler import NoiseScheduler
from models.UNET import UNetDiff

@dataclass
class DiffusionConfig:
    dataset_name: str
    learning_rate: int
    batch_size: int
    use_augmentation: bool
    timesteps_diff: int

@dataclass
class ModelConfig:
    pass


def train_diffusion(train_config, model_config):
    """ Main function to train the diffusion model """

    print(f"Training Config: {train_config}\n")
    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + "\n")

    # Load dataset
    train, test, dataset_information = get_dataset(train_config)
    print(f"Dataset information: {dataset_information}\n")

    dataloader_train = DataLoader(train, batch_size=train_config.batch_size, shuffle=True, drop_last=True, pin_memory=False)
    dataloader_test = DataLoader(test, batch_size=train_config.batch_size*3, pin_memory=False)

    model_config.image_size = dataset_information.image_size
    model_config.channels = dataset_information.channels
    model = UNetDiff(model_config)
    noise_scheduler = NoiseScheduler(train_config)

    for x_train, y_train in dataloader_train:
       
        x_train = x_train.to(device)
        image_with_noise = noise_scheduler.add_forward_noise(x_train)
        predicted_noise =  model(x_train)


        """
        grid = plot_grid_samples_tensor(x_train[:4], grid_size=[8,8])
        grid = plot_grid_samples_tensor(image_with_noise[:4], grid_size=[8,8])
        writer.add_image(f"train_sample, random", grid, 0)
        """
        exit()
        