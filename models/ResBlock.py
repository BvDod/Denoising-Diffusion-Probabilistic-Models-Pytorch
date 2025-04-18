import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, timestep_emb):
        """ 
        A residual block, with a timestep embedding added after the first conv
        """
        super(ResidualBlock, self).__init__()

        """
        self.timestemp_emb = nn.Sequential(
            timestep_emb,
            nn.Linear(timestemp_emb.dim, ch_in_out) # needs some reshaping still
        )
        """
        self.channels_in = ch_in
        self.channels_out = ch_out

        self.activation = nn.SiLU()
        self.normalize_1 = nn.GroupNorm(num_groups=32, num_channels=self.channels_in)

        self.conv_1 = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=3, padding=1)
        # time_emb added here
        self.normalize_2 = nn.GroupNorm(num_groups=32, num_channels=self.channels_out)

        self.conv_2 = nn.Conv2d(self.channels_out, self.channels_out, kernel_size=3, padding=1)

        if self.channels_in != self.channels_out:
            self.in_to_out_channels = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1)



    def forward(self, x):
        x_in = x
        x = self.activation(self.normalize_1(x))
        x = self.conv_1(x)
        # Add in time_emb here
        x = self.activation(self.normalize_2(x))
        x = self.conv_2(x)

        # Resize channels so they can be added together
        if self.channels_in != self.channels_out:
            x_in  = self.in_to_out_channels(x_in)
        
        return x + x_in
