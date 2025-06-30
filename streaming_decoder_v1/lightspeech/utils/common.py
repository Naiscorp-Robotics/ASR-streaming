import json
import wave
import contextlib
from typing import List, Tuple, Union, OrderedDict, Optional

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn.functional as F


def get_duration(audio_filepath: str):
    with contextlib.closing(wave.open(audio_filepath, "r")) as f:
        num_frames = f.getnframes()
        sample_rate = f.getframerate()
        duration = num_frames / sample_rate
    return duration


def load_dataset(filepaths: Union[str, List[str]]) -> List[dict]:
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dataset = []
    for filepath in filepaths:
        with open(filepath) as datas:
            dataset += [json.loads(d) for d in datas]

    return dataset


def compute_statistic(
    xs: torch.Tensor, x_lens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    masks = make_padding_mask(x_lens, xs.size(1))
    masks = masks[:, :, None].expand_as(xs)

    T = masks.sum(1)
    mean = (xs * masks).sum(1) / T
    std = (((xs - mean.unsqueeze(1)) ** 2 * masks).sum(1) / T).sqrt()

    return mean, std


def make_padding_mask(seq_lens: torch.Tensor, max_time: int) -> torch.Tensor:
    bs = seq_lens.size(0)
    device = seq_lens.device

    seq_range = torch.arange(0, max_time, dtype=torch.long, device=device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)

    seq_length = seq_lens.unsqueeze(-1)
    mask = seq_range < seq_length

    return mask


def length_regulator(
    xs: torch.Tensor,
    x_masks: torch.Tensor,
    durs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    y_lens = durs.sum(dim=1)
    y_masks = make_padding_mask(y_lens, y_lens.amax())

    b_x, t_x = x_masks.shape
    b_y, t_y = y_masks.shape
    assert b_x == b_y, "Batch size dimension isn't match"

    cum_durs = torch.cumsum(durs, 1).contiguous().view(b_x * t_x)
    alignment = make_padding_mask(cum_durs, t_y)
    alignment = alignment.contiguous().view(b_x, t_x, t_y).int()
    alignment = alignment - F.pad(alignment, [0, 0, 1, 0, 0, 0])[:, :-1]
    alignment = alignment * x_masks.unsqueeze(2) * y_masks.unsqueeze(1)
    alignment = alignment.type(xs.dtype)

    ys = torch.matmul(alignment.transpose(1, 2), xs)

    return ys, y_lens


def word_level_pooling(
    xs: torch.Tensor,
    wbs: torch.Tensor,
    reduction: Optional[str] = "sum",
) -> torch.Tensor:

    B, Tp, D = xs.size()

    Tw = wbs.amax() + 1
    wbs = wbs.masked_fill(wbs < 0, Tw)

    ys = xs.new_zeros([B, Tw + 1, D])
    ys = ys.scatter_add_(1, wbs[:, :, None].repeat([1, 1, D]), xs)
    ys = ys[:, :-1].contiguous()

    if reduction == "mean":
        ones = xs.new_ones(xs.shape[:2])
        N = xs.new_zeros([B, Tw + 1])
        N = N.scatter_add_(1, wbs, ones)
        N = N[:, :-1].contiguous()
        ys = ys / torch.clamp(N[:, :, None], min=1)

    return ys


def time_reduction(
    xs: torch.Tensor, x_lens: torch.Tensor, stride: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    b, t, d = xs.shape
    n = t + (stride - t % stride) % stride
    p = n - t

    xs = F.pad(xs, (0, 0, 0, p))
    xs = xs.reshape(b, n // stride, d * stride).contiguous()

    x_lens = torch.div(x_lens - 1, stride, rounding_mode="trunc")
    x_lens = (x_lens + 1).type(torch.long)

    return xs, x_lens


def load_module(
    hparams: DictConfig, weights: OrderedDict, device: Optional[str] = "cpu"
) -> torch.nn.Module:

    net = instantiate(hparams)
    net.load_state_dict(weights)
    net.to(device)

    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    return net
