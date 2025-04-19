import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.ResBlock import ResidualBlock
from models.transformer_block import TransformerBlock
import torch.nn.functional as F


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
        after_res1 = x

        x = self.resnet_block2(x)
        if self.use_transformer:
            x = self.transformer2(x)
        after_res2 = x

        if self.enable_downsample:
            x = self.conv_downsample(x)
        return x, after_res1, after_res2


class UpsampleBlock(nn.Module):
    def __init__(self, ch_in, ch_out, use_transformer):
        """ 
        """
        super(UpsampleBlock, self).__init__()

        self.use_transformer = use_transformer
        self.enable_upsample = True

        self.resnet_block1 = ResidualBlock(ch_in*2, ch_in, None)
        self.resnet_block2 = ResidualBlock(ch_in*2, ch_out, None)
        self.resnet_block3 = ResidualBlock(ch_out*2, ch_out, None)
        
        if self.use_transformer:
            self.transformer1 = TransformerBlock(ch_in)
            self.transformer2 = TransformerBlock(ch_out)
            self.transformer3 = TransformerBlock(ch_out)

        self.conv_upsample = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding= 1)
    
    def disable_upsample(self):
        self.enable_upsample = False
        self.conv_upsample = None
    

    def forward(self, x, skipped_xs):

        h1, h2, h3 = skipped_xs

        x = torch.concat([x, h3], dim=1)
        x = self.resnet_block1(x)
        if self.use_transformer:
            x = self.transformer1(x)

        x = torch.concat([x, h2], dim=1)
        x = self.resnet_block2(x)
        if self.use_transformer:
            x = self.transformer2(x)
        
        x = torch.concat([x, h1], dim=1)
        x = self.resnet_block3(x)
        if self.use_transformer:
            x = self.transformer3(x)

        if self.enable_upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.conv_upsample(x)
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
            for i in range(1, len(self.layer_channels))])
        self.downsample_layers[-1].disable_downsample() 

        # Middle
        channels = self.layer_channels[-1]
        self.middle_layers = nn.Sequential(
            ResidualBlock(channels, channels, None),
            TransformerBlock(channels),
            ResidualBlock(channels, channels, None),
        )

        # Upsampling
        layer_channels_up = list(reversed(self.layer_channels))
        layer_channels_up = layer_channels_up
        transformer_layers_up = list(reversed(self.transformer_layers_bool))
        self.layer_channels_up = layer_channels_up
        self.transformer_layers_up = transformer_layers_up

        self.upsample_layers = nn.Sequential(*[
            UpsampleBlock(layer_channels_up[i-1], layer_channels_up[i], transformer_layers_up[i-1]) 
            for i in range(1, len(layer_channels_up))])
        self.upsample_layers[-1].disable_upsample() 
    	
        # final head
        self.activation = nn.SiLU()
        self.normalize_1 = nn.GroupNorm(num_groups=32, num_channels=self.model_dim)
        self.conv_out = nn.Conv2d(self.model_dim, self.channels, kernel_size=1,)
        


    

    def calculate_resolution_layer_channels(self):
               
        self.dim_multiply = self.model_settings.dim_multiply # List of base dim multiplication for each layer 
        assert(self.resolution_amount == len(self.dim_multiply))

        self.dim_multiply = [1,] + list(self.dim_multiply) # add base channel dim
        layer_ch = [self.model_dim * mult for mult in self.dim_multiply] # Calculate actual channels   

        return layer_ch  


    def forward(self, x):
        x = self.resize_in(x)

        # Downsampling
        intermediate_xs = []
        for layer in self.downsample_layers:
            intermediate_xs.append(x)
            x, after_res1, after_res2 = layer(x)
            intermediate_xs.append(after_res1), intermediate_xs.append(after_res2)    

        # Middle
        x = self.middle_layers(x)

        # Upsampling
        for layer in self.upsample_layers:
            skipped_xs = intermediate_xs[-3:]
            del intermediate_xs[-3:]

            assert(len(skipped_xs) == 3)
            x = layer(x, skipped_xs)
        
        # Head
        x = self.activation(self.normalize_1(x))
        x = self.conv_out(x)
        return x
        


    
        