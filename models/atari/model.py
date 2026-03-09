from torch import nn
import torch
from collections import OrderedDict


class DeepQNetwork(nn.Module):
    """Deep Q Network for selecting actions based off states"""

    def __init__(self, img_height, img_width, action_space, kernel_size = 4):
        super().__init__()
        
        self.kernel_size = kernel_size
        self._dummy_pass = torch.rand(1, 1, img_height, img_width)

        self.conv_layers = nn.Sequential(
            OrderedDict([
                ('conv2d', nn.Conv2d(in_channels=1, out_channels=4, kernel_size=self.kernel_size)),
                ('silu', nn.SiLU()),
                ('conv2d2', nn.Conv2d(in_channels=4, out_channels=4, kernel_size=self.kernel_size)),
                ('silu2', nn.SiLU()),
                ('conv2d3', nn.Conv2d(in_channels=4, out_channels=1, kernel_size=self.kernel_size)),
                ('silu3', nn.SiLU()),
            ])    
        )

        _output_dummy = self.layers(self._dummy_pass)
        linear_dim = _output_dummy.shape[2] * _output_dummy.shape[3]

        self.ff_layers = nn.Sequential(
            OrderedDict([
                ('linear1', nn.Linear(linear_dim, linear_dim)),
                ('gelu', nn.GELU()),
                ('linear2', nn.Linear(linear_dim, action_space)),
                ('gelu2', nn.GELU())
            ])
        )

    def forward(self, x):
        # Pass through conv layers
        conv_pass = self.conv_layers(x)

        # Now flatten array
        flatten_array = torch.flatten(conv_pass, start_dim=1)

        action_vector = self.ff_layers(flatten_array)

        # Action_vector
        return action_vector
