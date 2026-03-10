from torch import nn
import torch
from collections import OrderedDict


class DeepQNetwork(nn.Module):
    """Deep Q Network for selecting actions based off states"""

    def __init__(self, img_height, img_width, action_space, num_frames, stride=2):
        super().__init__()
        
        self._dummy_pass = torch.rand(1, num_frames, img_height, img_width)

        self.conv_layers = nn.Sequential(
            OrderedDict([
                ('conv2d', nn.Conv2d(in_channels=num_frames, out_channels=8, kernel_size=8, stride=stride)),
                ('silu', nn.GELU()),
                ('conv2d2', nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=stride)),
                ('silu2', nn.GELU()),
                ('conv2d3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=stride)),
                ('silu3', nn.GELU()),
                ('conv2d4', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=stride)),
                ('silu4', nn.GELU()),
                ('conv2d5', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=stride)),
                ('silu5', nn.GELU()),
            ])    
        )

        _output_dummy = self.conv_layers(self._dummy_pass)
        linear_dim = _output_dummy.flatten(start_dim=1).shape[1]
        print(f"Linear Dim {linear_dim}")
        self.ff_layers = nn.Sequential(
            OrderedDict([
                ('linear1', nn.Linear(linear_dim, 512)),
                ('gelu', nn.GELU()),
                ('linear2', nn.Linear(512, linear_dim)),
                ('gelu2', nn.GELU()),
                ('linear3', nn.Linear(linear_dim, 512)),
                ('gelu3', nn.GELU()),
                ('action_linear', nn.Linear(512, action_space)),
            ])
        )

    def forward(self, x):
        # Pass through conv layers
        conv_pass = self.conv_layers(x)
        
        # Now flatten array
        flatten_array = torch.flatten(conv_pass, start_dim=1)
        
        # Action_vector
        action_vector = self.ff_layers(flatten_array)
        
        return action_vector
