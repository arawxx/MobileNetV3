import torch
import torch.nn as nn
from model.bottleneck import Bottleneck
from model.hard_activations import HSwish


class MobileNetV3(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_prob = 0.2):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            HSwish(),
        )

        self.bottlenecks = nn.Sequential(
            Bottleneck(input_channels=16, kernel=3, stride=1, expansion=16, output_channels=16, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=16, kernel=3, stride=2, expansion=64, output_channels=24, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=24, kernel=3, stride=1, expansion=72, output_channels=24, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=24, kernel=5, stride=2, expansion=72, output_channels=40, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=40, kernel=5, stride=1, expansion=120, output_channels=40, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=40, kernel=5, stride=1, expansion=120, output_channels=40, activation=nn.ReLU(inplace=True)),
            Bottleneck(input_channels=40, kernel=3, stride=2, expansion=240, output_channels=80, activation=HSwish()),
            Bottleneck(input_channels=80, kernel=3, stride=1, expansion=200, output_channels=80, activation=HSwish()),
            Bottleneck(input_channels=80, kernel=3, stride=1, expansion=184, output_channels=80, activation=HSwish()),
            Bottleneck(input_channels=80, kernel=3, stride=1, expansion=184, output_channels=80, activation=HSwish()),
            Bottleneck(input_channels=80, kernel=3, stride=1, expansion=480, output_channels=112, activation=HSwish()),
            Bottleneck(input_channels=112, kernel=3, stride=1, expansion=672, output_channels=112, activation=HSwish()),
            Bottleneck(input_channels=112, kernel=5, stride=2, expansion=672, output_channels=160, activation=HSwish()),
            Bottleneck(input_channels=160, kernel=5, stride=1, expansion=960, output_channels=160, activation=HSwish()),
            Bottleneck(input_channels=160, kernel=5, stride=1, expansion=960, output_channels=160, activation=HSwish()),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(960),
            HSwish(),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            HSwish(),
            nn.Dropout(p=dropout_prob, inplace=True),
            nn.Linear(1280, num_classes),
            # you may add your own final-layer activation function here, based on your use case
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bottlenecks(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
