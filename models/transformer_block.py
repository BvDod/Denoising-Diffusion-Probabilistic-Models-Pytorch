import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, channels):
        """ 
        """
        super(TransformerBlock, self).__init__()

        self.heads = 1
        self.channels = channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.channels)
        self.attention = nn.MultiheadAttention(channels, self.heads,
                                          dropout=0, batch_first=True)
        self.linear = nn.Linear(channels, channels)

    def forward(self, x_in):

        # Interstingly, no explicit positional encoding, should i try?

        b, c, h, w = x_in.shape

        x = self.norm1(x_in)

        x = torch.movedim(x, 1,-1)
        x = x.reshape((b,h*w,c))

        x = self.attention(x, x, x)[0]

        x = torch.movedim(x, -1,1)
        x = self.linear(x)
        x = x.reshape((b,c,h,w))

        x = x_in + x

        return x 