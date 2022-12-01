import torch.nn as nn


class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x *= self.relu6(x + 3) / 6
        return x


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.relu6(x + 3) / 6
        return x
