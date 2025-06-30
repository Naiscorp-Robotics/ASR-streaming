from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T_audio
from torch.nn.utils import weight_norm


LRELU_SLOPE = 0.2
SEGMENT_SIZE = 8192


class PWD(nn.Module):
    def __init__(self, period: int):
        super(PWD, self).__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 64, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(64, 128, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(128, 256, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(256, 512, (5, 1), (3, 1), (2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (5, 1), (1, 1), (2, 0))),
            ]
        )
        self.proj = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        b, c, t = xs.shape
        if t % self.period != 0:
            padding = self.period - (t % self.period)
            xs = F.pad(xs, (0, padding), "reflect")
            t = t + padding
        xs = xs.contiguous().view(b, c, t // self.period, self.period)

        fmap = []
        for layer in self.layers:
            xs = F.leaky_relu(layer(xs), LRELU_SLOPE)
            fmap.append(xs)

        xs = self.proj(xs)
        fmap.append(xs)

        xs = torch.flatten(xs, 1, -1)

        return xs, fmap


class MPWD(nn.Module):
    def __init__(self, periods: List[int]):
        super(MPWD, self).__init__()
        self.discriminators = nn.ModuleList([PWD(prd) for prd in periods])

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:

        outputs = [disc(xs) for disc in self.discriminators]
        disc_outs, fmap_outs = map(list, zip(*outputs))

        return disc_outs, fmap_outs


class RSD(nn.Module):
    def __init__(self, resolution: List[int]):
        super(RSD, self).__init__()
        self.spectrogram = T_audio.Spectrogram(*resolution, power=1)
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), (1, 1), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))),
            ]
        )
        self.proj = weight_norm(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        xs = self.spectrogram(xs)

        fmap = []
        for layer in self.layers:
            xs = F.leaky_relu(layer(xs), LRELU_SLOPE)
            fmap.append(xs)

        xs = self.proj(xs)
        fmap.append(xs)

        xs = torch.flatten(xs, 1, -1)

        return xs, fmap


class MRSD(nn.Module):
    def __init__(self, resolutions: List[List[int]]):
        super(MRSD, self).__init__()
        self.discriminators = nn.ModuleList([RSD(rst) for rst in resolutions])

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:

        outputs = [disc(xs) for disc in self.discriminators]
        disc_outs, fmap_outs = map(list, zip(*outputs))

        return disc_outs, fmap_outs


class PQMF(nn.Module):
    def __init__(
        self,
        N: int = 4,
        taps: int = 62,
        cutoff: float = 0.15,
        beta: float = 9.0,
    ):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta

        QMF = self._firwin(taps + 1, cutoff, beta)
        H = torch.zeros((N, len(QMF)))
        G = torch.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (
                (2 * k + 1)
                * (torch.pi / (2 * N))
                * (torch.arange(taps + 1) - ((taps - 1) / 2))
            )
            phase = (-1) ** k * torch.pi / 4

            H[k] = 2 * QMF * torch.cos(constant_factor + phase)
            G[k] = 2 * QMF * torch.cos(constant_factor - phase)

        H = H[:, None, :].float()
        G = G[None, :, :].float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N), dtype=torch.float)
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        outs = F.conv1d(xs, self.H, padding=self.taps // 2, stride=self.N)
        return outs

    def _firwin(
        self,
        numtaps: int,
        cutoff: float,
        beta: float,
    ) -> torch.Tensor:

        pi = torch.pi
        alpha = 0.5 * (numtaps - 1)

        coeff = torch.arange(0, numtaps) - alpha
        coeff = torch.sin(pi * cutoff * coeff) / (pi * cutoff * coeff)
        coeff = coeff.nan_to_num(1.0)

        window = torch.kaiser_window(numtaps, periodic=False, beta=beta)

        filters = cutoff * coeff * window
        filters /= filters.sum()

        return filters


class MDC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilations: List[int],
    ):
        super(MDC, self).__init__()

        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size * dilation - dilation) // 2,
                        dilation=dilation,
                    )
                )
            )

        self.proj = weight_norm(
            nn.Conv1d(out_channels, out_channels, 3, stride=stride, padding=1)
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:

        _xs = 0.0
        for layer in self.layers:
            _xs += layer(xs)

        xs = _xs / len(self.layers)

        xs = self.proj(xs)
        xs = F.leaky_relu(xs, LRELU_SLOPE)

        return xs


class SBD(nn.Module):
    def __init__(
        self,
        init_channel: int,
        channels: List[int],
        kernel: int,
        strides: List[int],
        dilations: List[List[int]],
    ):
        super(SBD, self).__init__()

        self.layers = nn.ModuleList()
        for c, s, d in zip(channels, strides, dilations):
            k = kernel
            self.layers.append(MDC(init_channel, c, k, s, d))
            init_channel = c

        self.proj = weight_norm(nn.Conv1d(init_channel, 1, 3, 1, 1))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        fmap = []
        for layer in self.layers:
            xs = layer(xs)
            fmap.append(xs)

        xs = self.proj(xs)
        fmap.append(xs)

        xs = torch.flatten(xs, 1, -1)

        return xs, fmap


class MSBD(torch.nn.Module):
    def __init__(
        self,
        time_kernels: List[int],
        freq_kernel: int,
        time_channels: List[int],
        freq_channels: List[int],
        time_strides: List[List[int]],
        freq_stride: List[int],
        time_dilations: List[List[List[int]]],
        freq_dilations: List[List[int]],
        time_subband: List[int],
    ):

        super(MSBD, self).__init__()

        self.N = 16
        self.M = 64

        self.time_subband_1 = time_subband[0]
        self.time_subband_2 = time_subband[1]
        self.time_subband_3 = time_subband[2]

        self.fsbd = SBD(
            init_channel=SEGMENT_SIZE // self.M,
            channels=freq_channels,
            kernel=freq_kernel,
            strides=freq_stride,
            dilations=freq_dilations,
        )

        self.tsbd1 = SBD(
            init_channel=time_subband[0],
            channels=time_channels,
            kernel=time_kernels[0],
            strides=time_strides[0],
            dilations=time_dilations[0],
        )

        self.tsbd2 = SBD(
            init_channel=time_subband[1],
            channels=time_channels,
            kernel=time_kernels[1],
            strides=time_strides[1],
            dilations=time_dilations[1],
        )

        self.tsbd3 = SBD(
            init_channel=time_subband[2],
            channels=time_channels,
            kernel=time_kernels[2],
            strides=time_strides[2],
            dilations=time_dilations[2],
        )

        self.pqmf_n = PQMF(N=self.N, taps=256, cutoff=0.03, beta=10.0)
        self.pqmf_m = PQMF(N=self.M, taps=256, cutoff=0.1, beta=9.0)

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:

        disc_outs, fmap_outs = [], []

        xm = self.pqmf_m(xs)
        xm = xm.transpose(2, 1)

        q4, fmap = self.fsbd(xm)
        disc_outs.append(q4)
        fmap_outs.append(fmap)

        xn = self.pqmf_n(xs)

        q3, fmap = self.tsbd3(xn[:, : self.time_subband_3, :])
        disc_outs.append(q3)
        fmap_outs.append(fmap)

        q2, fmap = self.tsbd2(xn[:, : self.time_subband_2, :])
        disc_outs.append(q2)
        fmap_outs.append(fmap)

        q1, fmap = self.tsbd1(xn[:, : self.time_subband_1, :])
        disc_outs.append(q1)
        fmap_outs.append(fmap)

        return disc_outs, fmap_outs


class MBD(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        groups: List[int],
    ):
        super(MBD, self).__init__()

        init_channel = 1
        self.layers = nn.ModuleList()
        for c, k, s, g in zip(channels, kernels, strides, groups):
            self.layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=init_channel,
                        out_channels=c,
                        kernel_size=k,
                        stride=s,
                        padding=(k - 1) // 2,
                        groups=g,
                    )
                )
            )
            init_channel = c

        self.proj = weight_norm(nn.Conv1d(channels[-1], 1, 3, 1, 1))

    def forward(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        fmap = []
        for layer in self.layers:
            xs = F.leaky_relu(layer(xs), LRELU_SLOPE)
            fmap.append(xs)

        xs = self.proj(xs)
        fmap.append(xs)

        xs = torch.flatten(xs, 1, -1)

        return xs, fmap


class MMBD(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernels: List[List[int]],
        strides: List[int],
        groups: List[int],
    ):
        super(MMBD, self).__init__()

        self.combd_1 = MBD(channels, kernels[0], strides, groups)
        self.combd_2 = MBD(channels, kernels[1], strides, groups)
        self.combd_3 = MBD(channels, kernels[2], strides, groups)

        self.pqmf_2 = PQMF(N=2, taps=256, cutoff=0.25, beta=10.0)
        self.pqmf_4 = PQMF(N=4, taps=192, cutoff=0.13, beta=10.0)

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:

        disc_outs, fmap_outs = [], []

        p3, fmap = self.combd_3(xs)
        disc_outs.append(p3)
        fmap_outs.append(fmap)

        x2_prime = self.pqmf_2(xs)[:, :1, :]
        p2, fmap = self.combd_2(x2_prime)
        disc_outs.append(p2)
        fmap_outs.append(fmap)

        x1_prime = self.pqmf_4(xs)[:, :1, :]
        p1, fmap = self.combd_1(x1_prime)
        disc_outs.append(p1)
        fmap_outs.append(fmap)

        return disc_outs, fmap_outs
