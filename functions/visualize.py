import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision

import torch

def plot_grid_samples_tensor(tensor, grid_size=[8,8]):
    """ Plots a grid of random samples from a tensor with grid size = grid size"""
    
    tensor = tensor.clone()  # clone to avoid modifying the original tensor
    tensor = tensor / 2 + 0.5  # unnormalize
    tensor = tensor.clamp(0, 1)  # clamp to [0, 1]
    grid = torchvision.utils.make_grid(tensor, nrow=grid_size[0])
    return grid