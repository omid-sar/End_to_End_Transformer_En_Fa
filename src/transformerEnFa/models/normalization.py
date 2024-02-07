import torch
import torch.nn as nn


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6 ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive 


    def forward(self, x):
        mean = x.mean( dim=-1,keepdim=True)
        std = x.std( dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
