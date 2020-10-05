import torch
from torch import nn
from einops import rearrange

class GSA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
