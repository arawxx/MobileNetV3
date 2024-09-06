import torch.nn as nn
from model.squeeze_excite import SqueezeExcite


class Bottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        stride: int,
        expansion: int,
        output_channels: int,
        activation: nn.Module,
        se: bool = False,
    ) -> None:
        """
        MobileNetV3 bottleneck block.

        Args:
            input_channels (`int`): Number of input channels.
            kernel (`int`): Convolution kernel size.
            stride (`int`): Convolution stride.
            expansion (`int`): Expansion size, indicating the middle layer's output channels.
            output_channels (`int`): Number of final output channels.
            activation (`nn.Module`): Activation function.
            se (`bool`, optional): Whether to use Squeeze-and-Excitation. Defaults to False.
        """
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
            SqueezeExcite(expansion) if se else nn.Identity(),
            
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
