from torch import nn
import torch
from collections import OrderedDict


class DeepQNetwork(nn.Module):
    """Deep Q Network for selecting actions based off states"""

    def __init__(self, kernel_size = 4):
        super().__init__()
        
        self.kernel_size = kernel_size

        self.layers = nn.Sequential(
            OrderedDict([
                'conv', nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.kernel_size),
                'silu', nn.SiLU()
            ])    
        )

    def forward(self, x):
        return self.layers(x)