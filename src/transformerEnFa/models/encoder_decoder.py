import torch
import torch.nn as nn
from transformerEnFa.models.blocks import EncoderBlock, DecoderBlock, FeedForwardBlock
from transformerEnFa.models.normalization import LayerNormalization


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask ):
        for layer in self.layers:
            x = layer( x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return self.proj(x)
