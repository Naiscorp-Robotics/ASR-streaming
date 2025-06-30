import os
import warnings

import numpy as np
import onnxruntime

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


torch.set_num_threads(1)
torch.set_grad_enabled(False)


def read_audio(path: str, sampling_rate: int = 16000):
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = T.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    return wav.squeeze(0)


def group_segments(segments, min_duration=3.0, max_duration=15.0):

    _segments = []
    for start, end in segments:
        dur = end - start
        if dur > max_duration:
            num_segs = round(dur / max_duration) + 1
            dur = dur / num_segs
            _segments += [(i * dur, (i + 1) * dur) for i in range(num_segs)]
        else:
            _segments.append((start, end))

    groups = [[]]
    duration = 0.0

    for start, end in _segments:
        if (duration + end - start) > max_duration:
            duration = end - start
            groups.append([(start, end)])
        else:
            duration += end - start
            groups[-1].append((start, end))

    last_duration = sum([(end - start) for start, end in groups[-1]])
    if (len(groups) > 1) and (last_duration < min_duration):
        groups = groups[:-2] + [groups[-2] + groups[-1]]

    return groups


class OnnxWrapper(object):
    def __init__(self, filepath):
        self.session = onnxruntime.InferenceSession(filepath)
        self.session.intra_op_num_threads = 1
        self.session.inter_op_num_threads = 1
        self.reset_states()

    def __call__(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            msg = f"Too many dimensions for input audio chunk {x.dim()}"
            raise ValueError(msg)

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[::step]
            sr = 16000

        if x.shape[0] > 1:
            raise ValueError("Onnx model does not support batching")

        if sr not in [16000]:
            raise ValueError(f"Supported sample rates: {[16000]}")

        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        ort_inputs = {"input": x.numpy(), "h0": self._h, "c0": self._c}
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs

        out = torch.tensor(out).squeeze(2)[:, 1]

        return out

    def reset_states(self):
        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")


class SileroVAD(object):
    def __init__(self, model_filepath: str):
        __, ext_type = os.path.splitext(model_filepath)
        if ext_type == ".onnx":
            self.model = OnnxWrapper(model_filepath)
        else:
            self.model = torch.jit.load(model_filepath, map_location="cpu")
            self.model.eval()

    def __call__(
        self,
        audio: torch.Tensor,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 1536,
        speech_pad_ms: int = 30,
        return_seconds: bool = True,
    ):

        """
        This method is used for splitting long audios
        into speech chunks using silero VAD model.

        Parameters
        ----------
        audio: torch.Tensor, one dimensional
            One dimensional float torch.Tensor,
            other types are casted to torch tensor if possible.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates.

        threshold: float (default - 0.5)
            Silero VAD outputs speech probabilities for each audio chunk,
            probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately,
            but "lazy" 0.5 is pretty good for most datasets.

        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk,
            wait for min_silence_duration_ms before separating it.

        window_size_samples: int (default - 1536 samples)
            Audio chunks of window_size_samples size are fed to the model.
            Silero VAD models were trained using 512, 1024, 1536 samples
            for 16k sample rate and 256, 512, 768 samples for 8k sample rate.
            Values other than these may affect model perfomance!!

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side

        return_seconds: bool (default - True)
            whether return timestamps in seconds (default - samples)

        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks
        """

        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except Exception:
                msg = "Audio cannot be casted to tensor. Cast it manually"
                raise TypeError(msg)

        if len(audio.shape) > 1:
            for i in range(len(audio.shape)):
                audio = audio.squeeze(0)
            if len(audio.shape) > 1:
                raise ValueError(
                    "More than one dimension in audio. \
                    Are you trying to process audio with 2 channels?"
                )

        if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            sampling_rate = 16000
            audio = audio[::step]
            warnings.warn(
                "Sampling rate is a multiply of 16000. \
                Casting to 16000 manually!"
            )
        else:
            step = 1

        if sampling_rate != 8000 and sampling_rate != 16000:
            warnings.warn(
                "Currently silero VAD models only support 8k/16k sample rates"
            )

        if sampling_rate == 8000 and window_size_samples > 768:
            warnings.warn(
                "window_size_samples is too big for 8000 sampling_rate! \
                Better set to 256, 512 or 768 for 8k sample rate!"
            )
        if window_size_samples not in [256, 512, 768, 1024, 1536]:
            warnings.warn(
                "Unusual window_size_samples! Supported window_size_samples:\n\
                - [512, 1024, 1536] for 16000 sampling_rate\n \
                - [256, 512, 768] for 8000 sampling_rate"
            )

        self.model.reset_states()
        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000

        audio_length = len(audio)

        speech_probs = []
        for offset in range(0, audio_length, window_size_samples):
            chunk = audio[offset: offset + window_size_samples]
            if len(chunk) < window_size_samples:
                padding = int(window_size_samples - len(chunk))
                chunk = F.pad(chunk, (0, padding))
            speech_prob = self.model(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)

        triggered = False
        speeches = []
        current_speech = {}
        neg_threshold = threshold - 0.15
        temp_end = 0

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech["start"] = window_size_samples * i
                continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech["end"] = temp_end
                    if (
                        current_speech["end"] - current_speech["start"]
                    ) > min_speech_samples:
                        speeches.append(current_speech)
                    temp_end = 0
                    current_speech = {}
                    triggered = False
                    continue

        if (
            current_speech
            and (audio_length - current_speech["start"]) > min_speech_samples
        ):
            current_speech["end"] = audio_length
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                start = speech["start"] - speech_pad_samples
                speech["start"] = int(max(0, start))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    start = speeches[i + 1]["start"] - silence_duration // 2
                    speeches[i + 1]["start"] = int(max(0, start))
                else:
                    speech["end"] += int(speech_pad_samples)
            else:
                end = speech["end"] + speech_pad_samples
                speech["end"] = int(min(audio_length, end))

        if return_seconds:
            for speech_dict in speeches:
                start = speech_dict["start"] / sampling_rate
                end = speech_dict["end"] / sampling_rate
                speech_dict["start"] = round(start, 2)
                speech_dict["end"] = round(end, 2)
        elif step > 1:
            for speech_dict in speeches:
                speech_dict["start"] *= step
                speech_dict["end"] *= step

        return speeches
