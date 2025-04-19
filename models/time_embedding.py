import torch.nn as nn
import math
import torch

class SinusodialEmbedding(nn.Module):
    def __init__(self, dim):
        """ 
        """
        self.dim = dim
        super(SinusodialEmbedding, self).__init__()

    
    def forward(self, timesteps):

        # Based on: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        # Note: slightly different than "Attention is all you need" 

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(0,half_dim, device="cuda") * -emb)
        emb = emb.repeat(timesteps.shape[0], 1)
        emb = timesteps.unsqueeze(1) * emb  
        emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
        return emb