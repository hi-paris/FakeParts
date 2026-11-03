import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, h_sizes, output_size):
        super(MLP, self).__init__()
        self.input = nn.Linear(input_size, h_sizes[0])
        self.hidden = nn.ModuleList([nn.Linear(h_sizes[i], h_sizes[i+1]) for i in range(len(h_sizes) - 1)])
        self.output = nn.Linear(h_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu(self.input(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.output(x)
        return x

