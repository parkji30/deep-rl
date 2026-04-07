from torch import nn
import torch


class DeepQNetwork(nn.Module):
    """Deep Q Network for selecting actions based off states"""

    def __init__(self, img_height, img_width, action_space, num_frames):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_frames, img_height, img_width)
            feature_dim = self.features(dummy).flatten(start_dim=1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)