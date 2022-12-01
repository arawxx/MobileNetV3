import torch.nn as nn
from squeeze_excite import SqueezeExcite


class Bottleneck(nn.Module):
    def __init__(self, input_channels, kernel, stride, expansion, activation):
        super().__init__()

        self.bottleneck = nn.Module(
            # expansion
            nn.Conv2d(input_channels, expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expansion),
            activation(),

            # depth-wise convolution
            nn.Conv2d(expansion, expansion, kernel_size=kernel, stride=stride, padding=kernel//2, groups=expansion, bias=False),
            nn.BatchNorm2d(expansion),
            activation(),

            # squeeze-and-excite
            SqueezeExcite(expansion),
            
            # point-wise convolution
            nn.Conv2d(input_channels, expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expansion),
            activation(),
        )

    def forward(self, x):
        x += self.bottleneck(x)
        return x
