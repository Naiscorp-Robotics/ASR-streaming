import torch
import torch.nn as nn
import torch.nn.functional as F

from lightspeech.layers.attention import MultiHeadSelfAttention
from lightspeech.layers.normalization import ScaleBiasNorm


class SqueezeformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        attn_num_heads: int,
        attn_group_size: int,
        attn_max_pos_encoding: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super(SqueezeformerBlock, self).__init__()

        self.attn = AttentionBlock(
            d_model=d_model,
            num_heads=attn_num_heads,
            group_size=attn_group_size,
            max_pos_encoding=attn_max_pos_encoding,
            dropout=dropout,
        )
        self.norm_attn = nn.LayerNorm(d_model)

        self.ffn1 = FeedForwardBlock(
            d_model=d_model,
            dropout=dropout,
        )
        self.norm_ffn1 = nn.LayerNorm(d_model)

        self.conv = ConvolutionBlock(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.norm_conv = nn.LayerNorm(d_model)

        self.ffn2 = FeedForwardBlock(
            d_model=d_model,
            dropout=dropout,
        )
        self.norm_ffn2 = nn.LayerNorm(d_model)

    def forward(
        self,
        xs: torch.Tensor,
        attn_masks: torch.Tensor,
        conv_masks: torch.Tensor,
    ) -> torch.Tensor:

        residual = xs.clone()
        xs = self.attn(xs, attn_masks)
        xs = xs + residual
        xs = self.norm_attn(xs)

        residual = xs.clone()
        xs = self.ffn1(xs)
        xs = xs + residual
        xs = self.norm_ffn1(xs)

        residual = xs.clone()
        xs = self.conv(xs, conv_masks)
        xs = xs + residual
        xs = self.norm_conv(xs)

        residual = xs.clone()
        xs = self.ffn2(xs)
        xs = xs + residual
        xs = self.norm_ffn2(xs)

        return xs


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = ScaleBiasNorm(d_model)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.pre_norm(xs)

        xs = self.linear1(xs)
        xs = self.activation(xs)
        xs = self.dropout(xs)

        xs = self.linear2(xs)
        xs = self.dropout(xs)

        return xs


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        group_size: int,
        max_pos_encoding: int,
        dropout: float,
    ):
        super(AttentionBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(
            d_model,
            num_heads,
            group_size,
            max_pos_encoding,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = ScaleBiasNorm(d_model)

    def forward(self, xs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        xs = self.pre_norm(xs)

        xs = self.mhsa(xs, xs, xs, masks)
        xs = self.dropout(xs)

        return xs


class ConvolutionBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super(ConvolutionBlock, self).__init__()
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = ScaleBiasNorm(d_model)

    def forward(self, xs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        xs = self.pre_norm(xs)

        xs = xs.transpose(1, 2)
        xs = self.pointwise_conv1(xs)
        xs = F.silu(xs)

        masks = masks.unsqueeze(1)
        xs = xs.masked_fill(masks, 0.0)

        xs = self.depthwise_conv(xs)
        xs = self.norm(xs)
        xs = F.silu(xs)

        xs = self.pointwise_conv2(xs)
        xs = xs.transpose(1, 2)
        xs = self.dropout(xs)

        return xs
