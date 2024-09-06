import torch.nn as nn


class HSwish(nn.Module):
    def __init__(self):
        """Hard Swish activation function."""
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = x * self.relu6(x + 3) / 6
        return x


class HSigmoid(nn.Module):
    def __init__(self):
        """Hard Sigmoid activation function."""
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.relu6(x + 3) / 6
        return x
