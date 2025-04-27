from dataclasses import dataclass
from functions.dataHandling import get_dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functions.visualize import plot_grid_samples_tensor
import torch
from functions.noise_scheduler import NoiseScheduler
from models.UNET import UNetDiff

from functions.log_to_tb import log_to_tensorboard

from torchinfo import summary
import time
import os

import copy

@dataclass
class DiffusionConfig:
    dataset_name: str
    learning_rate: int
    batch_size: int
    use_augmentation: bool
    timesteps_diff: int

@dataclass
class ModelConfig:
    lowest_resolution_size: int
    dim_multiply: list[int]
    transformer_layers: list[int]
    base_dim: int


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
    dataloader_test = DataLoader(test, batch_size=train_config.batch_size*2, pin_memory=False)

    model_config.image_size = dataset_information.image_size
    model_config.channels = dataset_information.channels
    model = UNetDiff(model_config).to(device)
    model.load_state_dict(torch.load("models/saved_models/model_3.pt", weights_only=True))

    ema_model = copy.deepcopy(model)
    model.load_state_dict(torch.load("models/saved_models/model_ema_3.pt", weights_only=True))

    def update_ema_variables(model, ema_model, ema_decay=0.999):
        with torch.no_grad():
            model_state_dict = model.state_dict()
            ema_state_dict = ema_model.state_dict()

            for key in model_state_dict.keys():
                ema_state_dict[key].mul_(ema_decay).add_(model_state_dict[key], alpha=1 - ema_decay)

    #print(summary(model, input_size=(32, 3, 128, 128)))


    noise_scheduler = NoiseScheduler(train_config)
    
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=train_config.learning_rate, params=model.parameters())
    scaler = torch.amp.GradScaler("cuda" ,enabled=True)

    for epoch in range(1000):
        print(f"\nStart Epoch{epoch}")

        # Training
        time_epoch_start = time.time()
        loss_epoch = []

        model.train()
        for i, (x_train, y_train) in enumerate(dataloader_train):
            images_to_log, metrics_to_log = {}, {}

            x_train = x_train.to(device)

            with torch.no_grad():
                image_with_noise, noise, timesteps = noise_scheduler.add_forward_noise(x_train)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                predicted_noise =  model(image_with_noise, timesteps)
                loss = loss_function(predicted_noise, noise)
                loss_epoch.append(loss.item())
                loss = loss / 4 

            scaler.scale(loss).backward()
            
            if (i + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                update_ema_variables(model, ema_model)

            if (i + 1) % 100 == 0:
                log_to_tensorboard(writer, {}, {"Loss 100 epochs": sum(loss_epoch[-99:])/99}, int((i + 1)/ 100) + (epoch*(len(dataloader_train) // 100)))


        images_to_log = {
        "Image Clean (Training)": x_train[-9:], 
        "Image with noise (Training)": image_with_noise[-9:],
        "Noise (Training)": noise[-9:],
        "Predicted Noise (Training)": predicted_noise[-9:] 
        }
        metrics_to_log = {"Loss (Training)": sum(loss_epoch)/len(loss_epoch)}
        log_to_tensorboard(writer, images_to_log, metrics_to_log, epoch)

        ## Save model to disk
        path = f"models/saved_models/"
        os.makedirs(path, exist_ok = True) 
        torch.save(model.state_dict(), path + f"model_{epoch}.pt")
        torch.save(ema_model.state_dict(), path + f"model_ema_{epoch}.pt")

        # Validation
        model.eval()
        optimizer.zero_grad()

        loss_epoch_val = []
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(dataloader_test):
                images_to_log, metrics_to_log = {}, {}

                x_test = x_test.to(device)

                with torch.no_grad():
                    image_with_noise, noise, timesteps = noise_scheduler.add_forward_noise(x_test)

                with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                    predicted_noise =  ema_model(image_with_noise, timesteps)
                    loss = loss_function(predicted_noise, noise)
                loss_epoch_val.append(loss.item())
   
        images_to_log = {
        "Image Clean (Eval)": x_test[-9:], 
        "Image with noise (Eval)": image_with_noise[-9:],
        "Noise (Eval)": noise[-9:],
        "Predicted Noise (Eval)": predicted_noise[-9:] 
        }
        metrics_to_log = {"Loss (Eval)": sum(loss_epoch_val)/len(loss_epoch_val)}
        log_to_tensorboard(writer, images_to_log, metrics_to_log, epoch)
        
        

        
        

        """
        grid = plot_grid_samples_tensor(x_train[:4], grid_size=[8,8])
        grid = plot_grid_samples_tensor(image_with_noise[:4], grid_size=[8,8])
        writer.add_image(f"train_sample, random", grid, 0)
        """
        