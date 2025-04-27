from torch.utils.tensorboard import SummaryWriter
import torch
from functions.noise_scheduler import NoiseScheduler
from models.UNET import UNetDiff

from functions.log_to_tb import log_to_tensorboard
from functions.visualize import plot_grid_samples_tensor

import matplotlib.pyplot as plt
from torchinfo import summary
import time
import os


def sample_image(train_config, model_config):
    """ Main function to train the diffusion model """


    print(f"Training Config: {train_config}\n")
    writer = SummaryWriter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + "\n")


    model_config.image_size = [128, 128]
    model_config.channels = 3

    # Load model
    model = UNetDiff(model_config).to(device)
    model.load_state_dict(torch.load("model_56.pt", weights_only=True))
    noise_scheduler = NoiseScheduler(train_config)
    print(sum(p.numel() for p in model.parameters()))
    model.eval()
    x = torch.randn([20, 3]+model_config.image_size, device=device)

    with torch.no_grad():
        for t in reversed(range(1000)):
            t_batch = torch.full((x.shape[0],), t, device=device)
            predicted_noise = model(x, t_batch)
            x = noise_scheduler.reverse_noise(x, predicted_noise, t)
            image = plot_grid_samples_tensor(x)
            # Convert the grid tensor to a NumPy array
            np_grid = image.permute(1, 2, 0).cpu().numpy()

            # Plot and save
            plt.imsave(f'images/{t}.png', np_grid)
