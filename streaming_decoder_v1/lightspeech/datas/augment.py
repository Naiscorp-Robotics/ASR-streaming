import random
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from lightspeech.utils.common import load_dataset
from lightspeech.utils.operation import fft_convolution


class OverlappedSpeechSimulation(object):
    def __init__(
        self,
        speech_filepath_8k: str = None,
        speech_filepath_16k: str = None,
        min_energy_ratio: float = -5.0,
        max_energy_ratio: float = 5.0,
        probability: Optional[float] = 0.2,
    ):
        self.probability = probability
        self.energy_ratio = torch.distributions.Uniform(
            min_energy_ratio, max_energy_ratio
        )

        if speech_filepath_8k:
            self.speech_8k = load_dataset(speech_filepath_8k)
        if speech_filepath_16k:
            self.speech_16k = load_dataset(speech_filepath_16k)

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        if int(sample_rate) == 8000 and hasattr(self, "speech_8k"):
            speech_dataset = self.speech_8k
        elif int(sample_rate) == 16000 and hasattr(self, "speech_16k"):
            speech_dataset = self.speech_16k
        else:
            return speech

        primary_speech = speech.clone()
        primary_length = primary_speech.size(1)

        secondary_speech = random.choice(speech_dataset)
        energy_ratio = self.energy_ratio.sample()

        secondary_filepath = secondary_speech["audio_filepath"]
        secondary_speech, __ = torchaudio.load(secondary_filepath)
        secondary_length = secondary_speech.size(1)

        mixing_length = random.randrange(1, primary_length // 2 + 1)
        mixing_length = min(secondary_length - 1, mixing_length)

        primary_start = random.randrange(0, primary_length - mixing_length)
        secondary_start = random.randrange(0, secondary_length - mixing_length)

        energy_primary = speech.square().mean().sqrt()
        energy_secondary = secondary_speech.square().mean().sqrt()
        coefficient = (10 ** (energy_ratio / 10)).sqrt()
        mixing_scale = energy_primary / (coefficient * energy_secondary + 1e-9)

        noisy_speech = secondary_speech[
            :, secondary_start : secondary_start + mixing_length
        ]
        primary_speech[:, primary_start : primary_start + mixing_length] += (
            mixing_scale * noisy_speech
        )

        return primary_speech


class ApplyImpulseResponse(object):
    def __init__(
        self,
        rir_filepath_8k: str = None,
        rir_filepath_16k: str = None,
        second_before_peak: Optional[float] = 0.01,
        second_after_peak: Optional[float] = 0.5,
        probability: Optional[float] = 0.2,
    ):
        self.probability = probability
        self.second_before_peak = second_before_peak
        self.second_after_peak = second_after_peak

        if rir_filepath_8k:
            self.rir_8k = load_dataset(rir_filepath_8k)
        if rir_filepath_16k:
            self.rir_16k = load_dataset(rir_filepath_16k)

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        if int(sample_rate) == 8000 and hasattr(self, "rir_8k"):
            rir_dataset = self.rir_8k
        elif int(sample_rate) == 16000 and hasattr(self, "rir_16k"):
            rir_dataset = self.rir_16k
        else:
            return speech

        rir_data = random.choice(rir_dataset)
        rir_filepath = rir_data["audio_filepath"]
        rir, sample_rate = torchaudio.load(rir_filepath)

        peak_index = rir.argmax()
        start_index = int(peak_index - self.second_before_peak * sample_rate)
        end_index = int(peak_index + self.second_after_peak * sample_rate)
        start_index = max(0, start_index)
        end_index = min(rir.size(1), end_index)

        rir = rir[:, start_index:end_index]
        rir /= rir.norm() + 1e-9
        rir = torch.flip(rir, [1])
        rir = rir[None, ...]

        padded_speech = F.pad(speech, (rir.size(2) - 1, 0))
        padded_speech = padded_speech[None, ...]

        reverbed_speech = fft_convolution(padded_speech, rir)[0]
        reverbed_speech *= speech.norm() / (reverbed_speech.norm() + 1e-9)
        reverbed_speech = reverbed_speech.clamp(-1.0, 1.0)

        return reverbed_speech


class AddBackgroundNoise(object):
    def __init__(
        self,
        noise_filepath_8k: str = None,
        noise_filepath_16k: str = None,
        min_snr_db: Optional[float] = 0.0,
        max_snr_db: Optional[float] = 30.0,
        probability: Optional[float] = 0.2,
    ):
        self.probability = probability
        self.snr_db = torch.distributions.Uniform(min_snr_db, max_snr_db)

        if noise_filepath_8k:
            self.noise_8k = load_dataset(noise_filepath_8k)
        if noise_filepath_16k:
            self.noise_16k = load_dataset(noise_filepath_16k)

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        if int(sample_rate) == 8000 and hasattr(self, "noise_8k"):
            noise_dataset = self.noise_8k
        elif int(sample_rate) == 16000 and hasattr(self, "noise_16k"):
            noise_dataset = self.noise_16k
        else:
            return speech

        noise_data = random.choice(noise_dataset)
        noise_filepath = noise_data["audio_filepath"]
        noise_duration = noise_data["duration"]

        speech_duration = speech.size(1) / sample_rate
        mismatch = int((noise_duration - speech_duration) * sample_rate)
        if mismatch > 0:
            frame_offset = random.randint(0, mismatch)
            noise, __ = torchaudio.load(
                noise_filepath,
                frame_offset=frame_offset,
                num_frames=speech.size(1),
            )
            rms_noise = noise.square().mean().sqrt() + 1e-9
        else:
            noise, __ = torchaudio.load(noise_filepath)
            rms_noise = noise.square().mean().sqrt() + 1e-9
            frame_offset = random.randint(0, -mismatch)
            noise = F.pad(noise, (frame_offset, -mismatch - frame_offset))

        snr_db = self.snr_db.sample()
        rms_speech = speech.square().mean().sqrt() + 1e-9
        scale = 10 ** (-snr_db / 20) * rms_speech / rms_noise

        noise = F.pad(noise, (0, speech.size(1) - noise.size(1)))
        noisy_speech = speech + scale * noise
        noisy_speech *= speech.norm() / (noisy_speech.norm() + 1e-9)
        noisy_speech = noisy_speech.clamp(-1.0, 1.0)

        return noisy_speech


class TimeMasking(object):
    def __init__(
        self,
        time_masks: Optional[int] = 10,
        time_width: Optional[float] = 0.05,
    ):
        self.time_masks = time_masks
        self.time_width = time_width
        self.augment = T.TimeMasking(1)

    def apply(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.unsqueeze(0)
        time_width = int(self.time_width * feature.size(-1))
        self.augment.mask_param = max(time_width, 1)
        for __ in range(self.time_masks):
            feature = self.augment(feature)
        return feature.squeeze(0)


class FrequencyMasking(object):
    def __init__(
        self,
        freq_masks: Optional[int] = 1,
        freq_width: Optional[int] = 27,
    ):
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.augment = T.FrequencyMasking(freq_width)

    def apply(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.unsqueeze(0)
        for __ in range(self.freq_masks):
            feature = self.augment(feature)
        return feature.squeeze(0)
