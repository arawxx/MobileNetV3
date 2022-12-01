import torch.nn as nn
from model.squeeze_excite import SqueezeExcite


class Bottleneck(nn.Module):
    def __init__(self, input_channels, kernel, stride, expansion, output_channels, activation):
        super().__init__()

        self.bottleneck = nn.Sequential(
            # expansion
            nn.Conv2d(input_channels, expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expansion),
            activation,

            # depth-wise convolution
            nn.Conv2d(expansion, expansion, kernel_size=kernel, stride=stride, padding=kernel//2, groups=expansion, bias=False),
            nn.BatchNorm2d(expansion),
            activation,

            # squeeze-and-excite
            SqueezeExcite(expansion),
            
            # point-wise convolution
            nn.Conv2d(expansion, output_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            activation,
        )
        
        # for residual skip connecting when the input size is different from output size
        self.downsample = None if input_channels == output_channels and stride == 1 else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )


    def forward(self, x):
        residual = x
        output = self.bottleneck(x)

        if self.downsample:
            residual = self.downsample(x)

        output = output + residual

        return output
