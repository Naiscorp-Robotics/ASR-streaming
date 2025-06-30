from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T_audio

from lightspeech.layers.block import SqueezeformerBlock
from lightspeech.utils.common import make_padding_mask, compute_statistic


class PredictorNetwork(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        d_model: int,
        dropout: float,
    ):
        super(PredictorNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn_layer = nn.GRU(embedding_dim, d_model, batch_first=True)
        self.rnn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        token_idxs: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        embs = self.embedding(token_idxs)

        outputs, states = self.rnn_layer(embs, states)
        outputs = self.rnn_norm(outputs)
        outputs = self.dropout(outputs)

        return outputs, states


class JointNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(JointNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        enc_outs: torch.Tensor,
        pred_outs: torch.Tensor,
    ) -> torch.Tensor:

        enc_outs = enc_outs.unsqueeze(2)
        pred_outs = pred_outs.unsqueeze(1)
        joint_outs = F.silu(enc_outs + pred_outs)
        outputs = self.linear(joint_outs)

        return outputs


class CTCDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(CTCDecoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, enc_outs: torch.Tensor) -> torch.Tensor:
        ctc_outs = F.silu(self.linear1(enc_outs))
        ctc_outs = self.linear2(ctc_outs)
        ctc_outs = ctc_outs.log_softmax(2)
        return ctc_outs


class WaveformDecoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        d_model: int,
        num_layers: int,
        attn_num_heads: int,
        attn_group_size: int,
        attn_max_pos_encoding: int,
        conv_kernel_size: int,
        dropout: float,
    ):
        super(WaveformDecoder, self).__init__()

        self.n_fft = n_fft
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            decoder_layer = SqueezeformerBlock(
                d_model=d_model,
                attn_num_heads=attn_num_heads,
                attn_group_size=attn_group_size,
                attn_max_pos_encoding=attn_max_pos_encoding,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            self.decoder_layers.append(decoder_layer)

        self.output_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=n_fft + 2,
            kernel_size=3,
            padding=1,
        )

        self.vocoder = T_audio.InverseSpectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length
        )

    def forward(
        self,
        xs: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        __, max_time, __ = xs.size()
        masks = make_padding_mask(x_lens, max_time)
        attn_masks = masks.unsqueeze(1).repeat([1, max_time, 1])
        attn_masks = attn_masks & attn_masks.transpose(1, 2)
        attn_masks = ~attn_masks
        conv_masks = ~masks

        for layer in self.decoder_layers:
            xs = layer(xs, attn_masks, conv_masks)

        xs = self.output_conv(xs.transpose(2, 1))
        mags, phases = xs.split(self.n_fft // 2 + 1, dim=1)

        reals = mags.exp() * phases.cos()
        imags = mags.exp() * phases.sin()
        xs = torch.stack([reals, imags], dim=-1)
        xs = torch.view_as_complex(xs.contiguous())
        xs = xs.masked_fill(conv_masks[:, None, :], 0.0)

        audio_outs = self.vocoder(xs)[:, None, :]
        audio_lens = (audio_outs.size(2) / xs.size(2) * x_lens).long()

        return audio_outs, audio_lens


class TemporalPoolingDecoder(nn.Module):
    def __init__(self, d_model: int):
        super(TemporalPoolingDecoder, self).__init__()
        self.linear1 = nn.Linear(2 * d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, xs: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        mean, std = compute_statistic(xs, x_lens)
        outs = torch.cat((mean, std), dim=1)

        outs = F.silu(self.linear1(outs))
        outs = self.linear2(outs)

        return outs
