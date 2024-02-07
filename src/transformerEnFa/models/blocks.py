import torch
import torch.nn as nn
from transformerEnFa.models.attention import MultiHeadAttentionBlock
from transformerEnFa.models.normalization import LayerNormalization


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias= True) # Define W1,B1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True) # Define W2, B2

    def forward(self, x):
        out = self.linear_1(x)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        return out
    

class ResidualConnection(nn.Module):

    def __init__(self,dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_msk, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_msk))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x