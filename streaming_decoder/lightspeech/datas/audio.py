from typing import List, Tuple

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torchaudio

def extract_filterbank(
    waveform: torch.Tensor,
    sample_rate: int,
    device: "cpu",
) -> torch.Tensor:

    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(0.05 * sample_rate),
        win_length=int(0.025 * sample_rate),
        hop_length=int(0.01 * sample_rate),
        n_mels=128,
        center=False,
    ).to(device)

    filterbank = transformation(waveform)
    filterbank = filterbank.clamp(1e-5).log()
    filterbank = torch.transpose(filterbank, 2, 1)
    lens = [x.size(0) for x in filterbank]
    lens = torch.tensor(lens, device=device)

    return filterbank, lens


def extract_spectrogram(
    waveform: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:

    transformation = torchaudio.transforms.Spectrogram(
        n_fft, win_length, hop_length, power=None
    )

    spectrogram = transformation(waveform)
    spectrogram = spectrogram.squeeze(0)

    return spectrogram


def inverse_spectrogram(
    spectrogram: torch.cfloat,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:

    transformation = torchaudio.transforms.InverseSpectrogram(
        n_fft, win_length, hop_length
    )

    waveform = transformation(spectrogram)
    waveform = waveform.unsqueeze(0)

    return waveform


def get_augmentation(config: DictConfig) -> Tuple[List[object]]:
    # audio augmentation
    augment_config = config.get("audio_augment", {})
    audio_augments = [instantiate(cfg) for __, cfg in augment_config.items()]

    # feature augmentation
    augment_config = config.get("feature_augment", {})
    feature_augments = [instantiate(cfg) for __, cfg in augment_config.items()]

    return audio_augments, feature_augments
