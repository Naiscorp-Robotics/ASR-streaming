import math
from typing import List, Optional, Tuple

import torch


__all__ = ["Emformer"]


def _get_activation_module(activation: str) -> torch.nn.Module:
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    elif activation == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation {activation}")


def _get_weight_init_gains(
    weight_init_scale_strategy: Optional[str], num_layers: int
) -> List[Optional[float]]:
    if weight_init_scale_strategy is None:
        return [None for _ in range(num_layers)]
    elif weight_init_scale_strategy == "depthwise":
        return [1.0 / math.sqrt(layer_idx + 1) for layer_idx in range(num_layers)]
    elif weight_init_scale_strategy == "constant":
        return [1.0 / math.sqrt(2) for layer_idx in range(num_layers)]
    else:
        raise ValueError(
            f"Unsupported weight_init_scale_strategy value {weight_init_scale_strategy}"
        )


class _EmformerAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        weight_init_gain: Optional[float] = None,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) is not a multiple of num_heads ({num_heads})."
            )

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf

        self.scaling = (self.input_dim // self.num_heads) ** -0.5

        self.emb_to_key_value = torch.nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.emb_to_query = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.out_proj = torch.nn.Linear(input_dim, input_dim, bias=True)

        if weight_init_gain:
            torch.nn.init.xavier_uniform_(
                self.emb_to_key_value.weight, gain=weight_init_gain
            )
            torch.nn.init.xavier_uniform_(
                self.emb_to_query.weight, gain=weight_init_gain
            )

    def masked_softmax(self, attention_weights_float, attention_mask):
        attention_weights_float = torch.exp(attention_weights_float)
        attention_weights_float = attention_weights_float.masked_fill(
            attention_mask.unsqueeze(0), 0
        )
        attention_weights_float = torch.div(
            attention_weights_float,
            torch.sum(attention_weights_float, -1).unsqueeze(-1).repeat(1, 1, 60),
        )
        return attention_weights_float

    def _gen_attention_probs(
        self,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attention_weights_float = attention_weights.float()
        attention_weights_float = attention_weights_float.masked_fill(
            attention_mask.unsqueeze(0), self.negative_inf
        )
        attention_probs = torch.nn.functional.softmax(
            attention_weights_float, dim=-1
        ).type_as(attention_weights)
        # attention_probs = self.masked_softmax(attention_weights_float,attention_mask )

        attention_probs = torch.nn.functional.dropout(
            attention_probs, p=float(self.dropout), training=self.training
        )
        return attention_probs

    def _forward_impl(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_val: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = utterance.size(1)
        T = right_context.size(0) + utterance.size(0) + summary.size(0)
        # Compute query with [right context, utterance, summary].
        query = self.emb_to_query(torch.cat([right_context, utterance, summary]))
        # Compute key and value with [mems, right context, utterance].
        key, value = self.emb_to_key_value(
            torch.cat([mems, right_context, utterance])
        ).chunk(chunks=2, dim=2)

        if left_context_key is not None and left_context_val is not None:
            right_context_blocks_length = right_context.size(0)
            key = torch.cat(
                [
                    key[: mems.size(0) + right_context_blocks_length],
                    left_context_key,
                    key[mems.size(0) + right_context_blocks_length :],
                ],
            )
            # print(right_context_blocks_length)
            value = torch.cat(
                [
                    value[: mems.size(0) + right_context_blocks_length],
                    left_context_val,
                    value[mems.size(0) + right_context_blocks_length :],
                ],
            )
        # Compute attention weights from query, key, and value.
        reshaped_query, reshaped_key, reshaped_value = [
            tensor.contiguous()
            .view(-1, B * self.num_heads, self.input_dim // self.num_heads)
            .transpose(0, 1)
            for tensor in [query, key, value]
        ]
        attention_weights = torch.bmm(
            reshaped_query * self.scaling, reshaped_key.transpose(1, 2)
        )

        # Compute padding mask.
        padding_mask = None
        # Compute attention probabilities.
        attention_probs = self._gen_attention_probs(
            attention_weights, attention_mask, padding_mask
        )

        # Compute attention.
        attention = torch.bmm(attention_probs, reshaped_value)
        # assert attention.shape == (
        #     B * self.num_heads,
        #     T,
        #     self.input_dim // self.num_heads,
        # )
        attention = attention.transpose(0, 1).contiguous().view(T, B, self.input_dim)

        # Apply output projection.
        output_right_context_mems = self.out_proj(attention)

        summary_length = summary.size(0)
        output_right_context = output_right_context_mems[: T - summary_length]
        output_mems = output_right_context_mems[T - summary_length :]
        if self.tanh_on_mem:
            output_mems = torch.tanh(output_mems)
        else:
            output_mems = torch.clamp(output_mems, min=-10, max=10)

        return output_right_context, output_mems, key, value

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        left_context_key: torch.Tensor,
        left_context_val: torch.Tensor,
        m_kv: torch.Tensor,
        m_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        query_dim = right_context.size(0) + utterance.size(0) + summary.size(0)
        key_dim = (
            right_context.size(0)
            + utterance.size(0)
            + mems.size(0)
            + left_context_key.size(0)
        )
        attention_mask = torch.zeros(query_dim, key_dim).to(
            dtype=torch.bool, device=utterance.device
        )
        attention_mask[-1, : mems.size(0)] = True
        attention_mask[:, : mems.size(0) - m_m] = True
        attention_mask[
            :,
            mems.size(0)
            + right_context.size(0) : mems.size(0)
            + right_context.size(0)
            + left_context_key.size(0)
            - m_kv,
        ] = True
        output, output_mems, key, value = self._forward_impl(
            utterance,
            right_context,
            summary,
            mems,
            attention_mask,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
        )
        return (
            output,
            output_mems,
            key[mems.size(0) + right_context.size(0) :],
            value[mems.size(0) + right_context.size(0) :],
        )


class _EmformerLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        segment_length: int,
        dropout: float = 0.0,
        activation: str = "relu",
        left_context_length: int = 0,
        max_memory_size: int = 0,
        weight_init_gain: Optional[float] = None,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.attention = _EmformerAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            weight_init_gain=weight_init_gain,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.memory_op = torch.nn.AvgPool1d(
            kernel_size=segment_length, stride=segment_length, ceil_mode=True
        )

        activation_module = _get_activation_module(activation)
        self.pos_ff = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, ffn_dim),
            activation_module,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, input_dim),
            torch.nn.Dropout(dropout),
        )
        self.layer_norm_input = torch.nn.LayerNorm(input_dim)
        self.layer_norm_output = torch.nn.LayerNorm(input_dim)

        self.left_context_length = torch.tensor(left_context_length).to(dtype=torch.int)
        self.segment_length = torch.tensor(segment_length).to(dtype=torch.int)
        self.max_memory_size = torch.tensor(max_memory_size).to(dtype=torch.int)
        self.input_dim = input_dim

        self.use_mem = max_memory_size > 0

    def _init_state(
        self, batch_size: int, device: Optional[torch.device]
    ) -> List[torch.Tensor]:
        empty_memory = torch.zeros(
            self.max_memory_size, batch_size, self.input_dim, device=device
        )
        left_context_key = torch.zeros(
            self.left_context_length, batch_size, self.input_dim, device=device
        )
        left_context_val = torch.zeros(
            self.left_context_length, batch_size, self.input_dim, device=device
        )
        past_length = torch.zeros(1, batch_size, dtype=torch.int32, device=device)
        return [empty_memory, left_context_key, left_context_val, past_length]

    def _unpack_state(
        self, state: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_length = state[3][0][0]
        past_left_context_length = torch.min(self.left_context_length, past_length).to(
            dtype=torch.int64
        )
        past_mem_length = torch.min(
            self.max_memory_size, torch.div(past_length, self.segment_length)
        ).to(dtype=torch.int64)
        pre_mems = state[0]
        lc_key = state[1]
        lc_val = state[2]
        # pre_mems = state[0][self.max_memory_size - past_mem_length :]
        # lc_key = state[1][self.left_context_length - past_left_context_length :]
        # lc_val = state[2][self.left_context_length - past_left_context_length :]
        m_m = past_mem_length
        m_kv = past_left_context_length
        return pre_mems, lc_key, lc_val, m_m, m_kv

    def _pack_state(
        self,
        next_k: torch.Tensor,
        next_v: torch.Tensor,
        update_length: int,
        mems: torch.Tensor,
        state: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        new_k = torch.cat([state[1], next_k])
        new_v = torch.cat([state[2], next_v])
        state[0] = torch.cat([state[0], mems])[-self.max_memory_size :]
        state[1] = new_k[new_k.shape[0] - self.left_context_length :]
        state[2] = new_v[new_v.shape[0] - self.left_context_length :]
        state[3] = state[3] + update_length
        return state

    def _process_attention_output(
        self,
        rc_output: torch.Tensor,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> torch.Tensor:
        result = self.dropout(rc_output) + torch.cat([right_context, utterance])
        result = self.pos_ff(result) + result
        result = self.layer_norm_output(result)
        return result

    def _apply_pre_attention_layer_norm(
        self, utterance: torch.Tensor, right_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_norm_input = self.layer_norm_input(torch.cat([right_context, utterance]))
        return (
            layer_norm_input[right_context.size(0) :],
            layer_norm_input[: right_context.size(0)],
        )

    def _apply_post_attention_ffn(
        self,
        rc_output: torch.Tensor,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rc_output = self._process_attention_output(rc_output, utterance, right_context)
        return rc_output[right_context.size(0) :], rc_output[: right_context.size(0)]

    def _apply_attention_infer(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        state: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if state is None:
            state = self._init_state(utterance.size(1), device=utterance.device)
        pre_mems, lc_key, lc_val, m_m, m_kv = self._unpack_state(state)
        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
            summary = summary[:1]
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        rc_output, next_m, next_k, next_v = self.attention(
            utterance=utterance,
            right_context=right_context,
            summary=summary,
            mems=pre_mems,
            left_context_key=lc_key,
            left_context_val=lc_val,
            m_kv=m_kv,
            m_m=m_m,
        )
        state = self._pack_state(next_k, next_v, utterance.size(0), mems, state)
        return rc_output, next_m, state

    def forward(
        self,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        mems: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:

        (
            layer_norm_utterance,
            layer_norm_right_context,
        ) = self._apply_pre_attention_layer_norm(utterance, right_context)
        rc_output, output_mems, output_state = self._apply_attention_infer(
            layer_norm_utterance, layer_norm_right_context, mems, state
        )
        output_utterance, output_right_context = self._apply_post_attention_ffn(
            rc_output, utterance, right_context
        )
        return output_utterance, output_right_context, output_state, output_mems


class _EmformerImpl(torch.nn.Module):
    def __init__(
        self,
        emformer_layers: torch.nn.ModuleList,
        segment_length: int,
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
    ):
        super().__init__()

        self.use_mem = max_memory_size > 0
        self.memory_op = torch.nn.AvgPool1d(
            kernel_size=segment_length,
            stride=segment_length,
            ceil_mode=True,
        )
        self.emformer_layers = emformer_layers
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size

    def forward(
        self,
        input: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None,
        left_context_key_states: Optional[torch.Tensor] = None,
        left_context_value_states: Optional[torch.Tensor] = None,
        update_length_states: Optional[torch.Tensor] = None,
    ):
        input = input.permute(1, 0, 2)
        right_context_start_idx = input.size(0) - self.right_context_length
        right_context = input[right_context_start_idx:]
        utterance = input[:right_context_start_idx]
        mems = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)

        output = utterance
        output_memory_states = torch.zeros_like(memory_states)
        output_left_context_key_states = torch.zeros_like(left_context_key_states)
        output_left_context_value_states = torch.zeros_like(left_context_value_states)
        output_update_length_states = torch.zeros_like(update_length_states)

        for layer_idx, layer in enumerate(self.emformer_layers):
            layer_states = list(
                [
                    memory_states[layer_idx],
                    left_context_key_states[layer_idx],
                    left_context_value_states[layer_idx],
                    update_length_states[layer_idx],
                ]
            )
            output, right_context, output_state, mems = layer(
                output,
                right_context,
                layer_states,
                mems,
            )
            output_memory_states[layer_idx] = output_state[0]
            output_left_context_key_states[layer_idx] = output_state[1]
            output_left_context_value_states[layer_idx] = output_state[2]
            output_update_length_states[layer_idx] = output_state[3]
        return (
            output.permute(1, 0, 2),
            output_memory_states,
            output_left_context_key_states,
            output_left_context_value_states,
            output_update_length_states,
        )


class Emformer(_EmformerImpl):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        segment_length: int,
        dropout: float = 0.0,
        activation: str = "relu",
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        weight_init_scale_strategy: Optional[str] = "depthwise",
        tanh_on_mem: bool = True,
        negative_inf: float = -1e8,
    ):
        weight_init_gains = _get_weight_init_gains(
            weight_init_scale_strategy, num_layers
        )
        emformer_layers = torch.nn.ModuleList(
            [
                _EmformerLayer(
                    input_dim,
                    num_heads,
                    ffn_dim,
                    segment_length,
                    dropout=dropout,
                    activation=activation,
                    left_context_length=left_context_length,
                    max_memory_size=max_memory_size,
                    weight_init_gain=weight_init_gains[layer_idx],
                    tanh_on_mem=tanh_on_mem,
                    negative_inf=negative_inf,
                )
                for layer_idx in range(num_layers)
            ]
        )
        super().__init__(
            emformer_layers,
            segment_length,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=max_memory_size,
        )
