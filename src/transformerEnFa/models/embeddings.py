import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncodding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create poition vector size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Calculate positional encoding to EVEN and ODD columns 
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (- math.log(10000.0)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension to tensor (seq_len, d_model) -> (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register "pe" in the buffer of this module:It ensures that the buffer's state is included 
        # when you save (torch.save) and load (torch.load) the model
        # Also we wont learn them , we calcualte once and then just we used it
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        return self.dropout(x)