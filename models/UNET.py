import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DownsampleBlock(nn.Module):
    def __init__(self, ch_in, ch_out, use_transformer):
        """ 
        """
        super(DownsampleBlock, self).__init__()

        self.use_transformer = use_transformer
        self.enable_downsample = True

        self.activation = nn.SiLU()
        self.resnet_block1 = ResidualBlock(ch_in, ch_out)
        self.resnet_block2 = ResidualBlock(ch_out, ch_out)
        
        if self.use_transformer:
            self.transformer = TransformerBlock(channels)

        self.conv_downsample = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding= 1)
    
    def disable_downsample(self):
        self.enable_downsample = False
        self.conv_downsample = None
    

    def forward(self, x):
        



class UNetDiff(nn.Module):
    """ """

    def __init__(self, model_settings):
        """ 
        """
        super(UNetDiff, self).__init__()
        self.model_settings = model_settings
        
        self.model_dim = 64 # Base embedding dim, where other dims are based on
        self.channels = self.model_settings.channels
        self.image_size = self.model_settings.image_size[0]
        self.lowest_resolution_size = self.model_settings.lowest_resolution_size

        self.time_embedding = nn.Sequential(
            sinusodialEmbedding(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim*4),
            torch.nn.SiLU(),
            nn.Linear(self.model_dim*4, self.model_dim*4),
        )
        
        # Amount of resolution levels
        # number of times we have to half input size to get lowest res we want
        self.resolution_amount = math.log2(self.image_size/self.lowest_resolution_size)
        self.dim_multiply = self.model_settings.dim_multiply # List of base dim multiplication for each layer 
        assert(self.resolution_amount == len(self.dim_multiply))
        self.dim_multiply = [1,] + self.dim_multiply # add base channel dim
        
        layer_ch = [self.model_dim * mult for mult in self.dim_multiply] # Calculate actuall channels   
        enable_transformer = [i in self.transformer_layers for i in range(self.resolution_amount)]

        self.resize_in = nn.conv2d(self.channels, self.model_dim, 1)
        
        self.downsample_layers = nn.Sequential(
            *[DownsampleBlock(layer_ch[i-1], layer_ch, enable_transformer[i]) for i in range(1, len(self.dim_multiply))]
        )
        self.downsample_layers[-1].disable_downsample()


    def forward(self, x):

        print(x.shape)

        return x