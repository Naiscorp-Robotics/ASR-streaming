from typing import Optional

import torch
import torch.nn as nn

from lightspeech.utils.common import time_reduction
from .emformer import Emformer


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
        num_heads: Optional[int] = 8,
        dropout: Optional[float] = 0.1,
        activation: Optional[str] = "gelu",
        max_memory_size: Optional[int] = 4,
        weight_init_scale_strategy: Optional[str] = "depthwise",
        tanh_on_mem: Optional[bool] = True,
    ):
        super(StreamingAcousticEncoder, self).__init__()

        self.stride = 4

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

    def forward(
        self,
        xs: torch.Tensor,
        x_lens: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None,
        left_context_key_states: Optional[torch.Tensor] = None,
        left_context_value_states: Optional[torch.Tensor] = None,
        update_length_states: Optional[torch.Tensor] = None,
    ):
        """Forward pass for streaming inference."""

        xs = self.input_linear(xs)
        xs, x_lens = time_reduction(xs, x_lens, self.stride)
        (
            xs,
            memory_states,
            left_context_key_states,
            left_context_value_states,
            update_length_states,
        ) = self.encoder_layers(
            xs,
            memory_states,
            left_context_key_states,
            left_context_value_states,
            update_length_states,
        )
        return (
            xs,
            x_lens,
            memory_states,
            left_context_key_states,
            left_context_value_states,
            update_length_states,
        )