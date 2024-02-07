import torch
import torch.nn as nn
from transformerEnFa.models.embeddings import InputEmbeddings, PositionalEncodding
from transformerEnFa.models.encoder_decoder import Encoder, Decoder, ProjectionLayer
from transformerEnFa.models.blocks import EncoderBlock, DecoderBlock, FeedForwardBlock, ResidualConnection
from transformerEnFa.models.attention import MultiHeadAttentionBlock



class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, projection_layer: ProjectionLayer,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncodding, tgt_pos: PositionalEncodding) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
    

    def encode(self, src, src_mask):
        # (Batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor):
         # (Batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
         # (Batch, seq_len, vocab_size)
        return self.projection_layer(x)


def built_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512,
                      N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncodding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncodding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    # Initilizing : Encoder-> EncoderBlock -> MultiHeadAttention/ FeedForwrd/ ResidualCoonection 
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer 
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer 
    transformer = Transformer(encoder, decoder, projection_layer, src_embed, tgt_embed, src_pos, tgt_pos)

    # Initialize the prameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


