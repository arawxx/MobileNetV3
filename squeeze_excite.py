import torch.nn as nn
from hard_activations import HSigmoid


class SqueezeExcite(nn.Module):
    def __init__(self, input_channels, squeeze = 4):
        super().__init__()

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(input_channels, out_channels=input_channels//squeeze, bias=False),
            nn.BatchNorm2d(input_channels//squeeze),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//squeeze, input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            HSigmoid(),
        )
    
    def forward(self, x):
        x *= self.SE(x)
        return x
