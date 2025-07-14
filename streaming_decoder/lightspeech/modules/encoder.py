from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Emformer

from lightspeech.layers.sampling import ConvolutionSubsampling
from lightspeech.layers.block import SqueezeformerBlock
from lightspeech.utils.common import (
    make_padding_mask,
    time_reduction,
    word_level_pooling,
    length_regulator,
)


class AcousticEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        subsampling_num_filters: int,
        subsampling_kernel_size: int,
        attn_num_heads: int,
        attn_group_size: int,
        attn_max_pos_encoding: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super(AcousticEncoder, self).__init__()

        self.subsampling = ConvolutionSubsampling(
            input_dim=input_dim,
            output_dim=d_model,
            num_filters=subsampling_num_filters,
            kernel_size=subsampling_kernel_size,
            dropout=dropout,
        )

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            encoder_layer = SqueezeformerBlock(
                d_model=d_model,
                attn_num_heads=attn_num_heads,
                attn_group_size=attn_group_size,
                attn_max_pos_encoding=attn_max_pos_encoding,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            self.encoder_layers.append(encoder_layer)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs, x_lens = self.subsampling(xs, x_lens)

        __, max_time, __ = xs.size()
        masks = make_padding_mask(x_lens, max_time)
        attn_masks = masks.unsqueeze(1).repeat([1, max_time, 1])
        attn_masks = attn_masks & attn_masks.transpose(1, 2)
        attn_masks = ~attn_masks
        conv_masks = ~masks

        for layer in self.encoder_layers:
            xs = layer(xs, attn_masks, conv_masks)

        return xs, x_lens


class StreamingAcousticEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        segment_length: int,
        left_context_length: int,
        right_context_length: int,
        ffn_dim: int,
        num_layers: int,
        subsampling_factor: int=4,
        num_heads: Optional[int] = 8,
        dropout: Optional[float] = 0.1,
        activation: Optional[str] = "gelu",
        max_memory_size: Optional[int] = 4,
        weight_init_scale_strategy: Optional[str] = "depthwise",
        tanh_on_mem: Optional[bool] = True,
    ):
        super(StreamingAcousticEncoder, self).__init__()

        self.stride = subsampling_factor

        assert (
            d_model % self.stride == 0
        ), "The model dimension must be divisible by the stride."

        self.input_linear = nn.Linear(
            in_features=input_dim,
            out_features=d_model // self.stride,
            bias=False,
        )
        self.encoder_layers = Emformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            segment_length=segment_length // self.stride,
            dropout=dropout,
            activation=activation,
            left_context_length=left_context_length // self.stride,
            right_context_length=right_context_length // self.stride,
            max_memory_size=max_memory_size,
            weight_init_scale_strategy=weight_init_scale_strategy,
            tanh_on_mem=tanh_on_mem,
        )

    @torch.jit.unused
    def forward(
        self,
        xs: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and non-streaming inference."""

        xs = F.pad(xs, (0, 0, 0, self.right_padding))
        xs = self.input_linear(xs)

        xs, x_lens = time_reduction(xs, x_lens, self.stride)
        xs, x_lens = self.encoder_layers(xs, x_lens)

        return xs, x_lens

    @torch.jit.export    
    def infer(
        self,
        xs: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Forward pass for streaming inference."""

        xs = self.input_linear(xs)

        xs, x_lens = time_reduction(xs, x_lens, self.stride)
        xs, x_lens, states = self.encoder_layers.infer(xs, x_lens, states)

        return xs, x_lens, states


class LinguisticEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        attn_num_heads: int,
        attn_group_size: int,
        attn_max_pos_encoding: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super(LinguisticEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.phoneme_encoder_layers = nn.ModuleList(
            [
                SqueezeformerBlock(
                    d_model=d_model,
                    attn_num_heads=attn_num_heads,
                    attn_group_size=attn_group_size,
                    attn_max_pos_encoding=attn_max_pos_encoding,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for __ in range(num_layers)
            ]
        )
        self.word_encoder_layers = nn.ModuleList(
            [
                SqueezeformerBlock(
                    d_model=d_model,
                    attn_num_heads=attn_num_heads,
                    attn_group_size=attn_group_size,
                    attn_max_pos_encoding=attn_max_pos_encoding,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for __ in range(num_layers)
            ]
        )
        self.w2p_attention = nn.MultiheadAttention(
            d_model,
            attn_num_heads,
            dropout,
            batch_first=True,
        )

        padding = (conv_kernel_size - 1) // 2
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=padding),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=padding),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, 1, 3, 1, 1),
        )

    def forward(
        self,
        token_idxs: torch.Tensor,
        token_lens: torch.Tensor,
        word_idxs: torch.Tensor,
        word_durs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:

        # Phoneme Embedding
        p_embs = self.embedding(token_idxs)

        # Phoneme Mask
        __, max_time, __ = p_embs.size()
        masks = make_padding_mask(token_lens, max_time)
        attn_masks = masks.unsqueeze(1).repeat([1, max_time, 1])
        attn_masks = attn_masks & attn_masks.transpose(1, 2)
        attn_masks = ~attn_masks
        conv_masks = ~masks

        # Phoneme Encoding
        p_enc_outs = p_embs.clone()
        for layer in self.phoneme_encoder_layers:
            p_enc_outs = layer(p_enc_outs, attn_masks, conv_masks)

        # Phoneme Duration Prediction
        p_durs = self.duration_predictor(p_enc_outs.transpose(1, 2))
        p_durs = p_durs.transpose(1, 2).squeeze(2)
        p_durs = p_durs.masked_fill(conv_masks, 0.0)

        # Word-level Pooling
        w_embs = word_level_pooling(p_enc_outs, word_idxs, reduction="mean")
        w_lens = word_idxs.amax(dim=1) + 1  # plus offset
        w_durs = word_level_pooling(p_durs.exp().unsqueeze(2), word_idxs)
        w_durs = w_durs.squeeze(2)

        # Word Mask
        __, max_time, __ = w_embs.size()
        masks = make_padding_mask(w_lens, max_time)
        attn_masks = masks.unsqueeze(1).repeat([1, max_time, 1])
        attn_masks = attn_masks & attn_masks.transpose(1, 2)
        attn_masks = ~attn_masks
        conv_masks = ~masks

        # Word Encoding
        w_enc_outs = w_embs.clone()
        for layer in self.word_encoder_layers:
            w_enc_outs = layer(w_enc_outs, attn_masks, conv_masks)

        if word_durs is None:
            word_durs = w_durs.ceil().long().clamp(min=10)
            word_durs = word_durs.masked_fill(conv_masks, 0)

        # Word-level Length Regulator
        w_enc_outs, w_enc_lens = length_regulator(w_enc_outs, masks, word_durs)

        # Word-to-Phoneme Attention
        key_padding_masks = make_padding_mask(token_lens, token_idxs.size(1))
        key_padding_masks = ~key_padding_masks
        w_enc_outs, __ = self.w2p_attention(
            w_enc_outs,
            p_enc_outs,
            p_enc_outs,
            key_padding_masks,
        )

        return w_enc_outs, w_enc_lens, w_durs
