import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.ResBlock import ResidualBlock
from models.transformer_block import TransformerBlock

class DownsampleBlock(nn.Module):
    def __init__(self, ch_in, ch_out, use_transformer, timestep_emb):
        """ 
        """
        super(DownsampleBlock, self).__init__()

        self.use_transformer = use_transformer
        self.enable_downsample = True
        self.timestep_emb = timestep_emb

        self.activation = nn.SiLU()
        self.resnet_block1 = ResidualBlock(ch_in, ch_out, self.timestep_emb)
        self.resnet_block2 = ResidualBlock(ch_out, ch_out, self.timestep_emb)
        
        if self.use_transformer:
            self.transformer1 = TransformerBlock(ch_out)
            self.transformer2 = TransformerBlock(ch_out)

        self.conv_downsample = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding= 1)
    
    def disable_downsample(self):
        self.enable_downsample = False
        self.conv_downsample = None
    

    def forward(self, x):
        x = self.resnet_block1(x)
        if self.use_transformer:
            x = self.transformer1(x)

        x = self.resnet_block2(x)
        if self.use_transformer:
            x = self.transformer2(x)
        
        if self.enable_downsample:
            x = self.conv_downsample(x)
        return x


        
        



class UNetDiff(nn.Module):
    """ """

    def __init__(self, model_settings):
        """ 
        """
        super(UNetDiff, self).__init__()
        self.model_settings = model_settings
        
        self.model_dim = model_settings.base_dim # Base embedding dim, where other dims are based on
        self.channels = model_settings.channels
        self.image_size = model_settings.image_size[0]
        self.lowest_resolution_size = model_settings.lowest_resolution_size
        self.transformer_layers = model_settings.transformer_layers
        
        # number of times we have to half input size to get lowest res we want
        self.resolution_amount = 1 + int(math.log2(self.image_size/self.lowest_resolution_size))
        self.layer_channels = self.calculate_resolution_layer_channels() # Get channel amount at each layer
        self.transformer_layers_bool = [i in self.transformer_layers for i in range(self.resolution_amount)]
        
        self.time_embedding = None
        """
        nn.Sequential(
            sinusodialEmbedding(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim*4),
            torch.nn.SiLU(),
            nn.Linear(self.model_dim*4, self.model_dim*4),
        )
        """
        
        # Downsampling 
        self.resize_in = nn.Conv2d(self.channels, self.model_dim, 1)
        self.downsample_layers = nn.Sequential(*[
            DownsampleBlock(self.layer_channels[i-1], self.layer_channels[i], self.transformer_layers_bool[i-1], self.time_embedding) 
            for i in range(1, len(self.dim_multiply))])
        self.downsample_layers[-1].disable_downsample() 


    

    def calculate_resolution_layer_channels(self):
               
        self.dim_multiply = self.model_settings.dim_multiply # List of base dim multiplication for each layer 
        print(self.resolution_amount)
        assert(self.resolution_amount == len(self.dim_multiply))

        self.dim_multiply = [1,] + list(self.dim_multiply) # add base channel dim
        layer_ch = [self.model_dim * mult for mult in self.dim_multiply] # Calculate actual channels   

        return layer_ch  


    def forward(self, x):
        print(x.shape)
        x = self.resize_in(x)
        x = self.downsample_layers(x)
        print(x.shape)
        return x