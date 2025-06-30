import random
from typing import Tuple, List, Union, Optional

from omegaconf import DictConfig

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from lightspeech.datas.text import build_vocab, build_lexicon, tokenize
from lightspeech.datas.audio import (
    get_augmentation,
    extract_filterbank,
    extract_spectrogram,
)
from lightspeech.utils.common import load_dataset, time_reduction


def collate_ssl_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:

    features = [b[0] for b in batch]
    feature_lengths = [len(f) for f in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    targets = [b[1] for b in batch]
    targets = pad_sequence(targets, batch_first=True)

    return features, feature_lengths, targets


def collate_asr_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:

    features = [b[0] for b in batch]
    feature_lengths = [len(f) for f in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    tokens = [b[1] for b in batch]
    token_lengths = [len(t) for t in tokens]
    tokens = pad_sequence(tokens, batch_first=True)
    token_lengths = torch.tensor(token_lengths, dtype=torch.long)

    return features, feature_lengths, tokens, token_lengths


def collate_tts_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:

    token_idxs = [b[0] for b in batch]
    token_lens = [len(idxs) for idxs in token_idxs]
    token_idxs = pad_sequence(token_idxs, batch_first=True)
    token_lens = torch.tensor(token_lens, dtype=torch.long)

    word_idxs = [b[1] for b in batch]
    word_idxs = pad_sequence(word_idxs, batch_first=True)

    word_durs = [b[2] for b in batch]
    word_durs = pad_sequence(word_durs, batch_first=True)

    audio_tgts = [b[3] for b in batch]
    audio_lens = [len(audio) for audio in audio_tgts]
    audio_tgts = pad_sequence(audio_tgts, batch_first=True).transpose(1, 2)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)

    return (
        token_idxs,
        token_lens,
        word_idxs,
        word_durs,
        audio_tgts,
        audio_lens,
    )


def collate_sc_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:

    features = [b[0] for b in batch]
    feature_lengths = [len(f) for f in features]
    features = pad_sequence(features, batch_first=True)
    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    targets = [b[1] for b in batch]
    targets = torch.stack(targets)

    return features, feature_lengths, targets


class SpeechRepresentationDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRepresentationDataset, self).__init__()

        self.framerate = 4  # subsampling factor of acoustic encoder
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        target = extract_filterbank(speech, sample_rate)

        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        target = target.t().unsqueeze(0)
        target_length = torch.tensor(target.size(1))
        target, __ = time_reduction(target, target_length, self.framerate)
        target = target.squeeze(0)

        # TODO
        mean = target.mean(dim=1, keepdim=True)
        std = target.std(dim=1, keepdim=True)
        target = (target - mean) / (std + 1e-9)

        return feature.t(), target

    def __len__(self) -> int:
        return len(self.dataset)


class SpeechRecognitionDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechRecognitionDataset, self).__init__()

        self.vocab = build_vocab()
        self.lexicon = build_lexicon()
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        transcript = data["transcript"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        tokens = tokenize(transcript, self.vocab, self.lexicon)
        tokens = [self.vocab.index(token) for token in tokens]
        tokens = torch.tensor(tokens, dtype=torch.long)

        return feature.t(), tokens

    def __len__(self) -> int:
        return len(self.dataset)


class SpeechSynthesisDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        n_fft: int,
        win_length: int,
        hop_length: int,
    ):
        super(SpeechSynthesisDataset, self).__init__()

        self.blank = 0
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.lexicon = build_lexicon()
        self.vocab = build_vocab()
        self.dataset = load_dataset(filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        audio_length = data["duration"]
        alignment = data["alignment"]

        speech, __ = torchaudio.load(audio_filepath)
        feature = extract_spectrogram(
            speech, self.n_fft, self.win_length, self.hop_length
        )
        feature_length = feature.size(1)

        token_idxs, word_idxs, word_durs = [], [], []
        for word, start, end in alignment:
            word_idx = max(word_idxs) + 1 if len(word_idxs) > 0 else 0

            if word == "":
                token_idxs += [self.blank]
                word_idxs += [word_idx]
            else:
                tokens = tokenize(word, self.vocab, self.lexicon)
                token_idxs += [self.vocabulary.index(t) + 1 for t in tokens]
                word_idxs += [word_idx] * len(tokens)

            word_dur = round((end - start) / audio_length * feature_length)
            word_durs.append(word_dur)

        mismatch = feature_length - sum(word_durs)
        bias = 1 if mismatch >= 0 else -1
        for __ in range(abs(mismatch)):
            idx = random.randrange(len(word_durs))
            word_durs[idx] += bias

        token_idxs = torch.tensor(token_idxs, dtype=torch.long)
        word_idxs = torch.tensor(word_idxs, dtype=torch.long)
        word_durs = torch.tensor(word_durs, dtype=torch.long)

        return token_idxs, word_idxs, word_durs, speech.t()

    def __len__(self) -> int:
        return len(self.dataset)


class SpeechClassificationDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        categories: List[str],
        augmentation: Optional[DictConfig] = None,
    ):
        super(SpeechClassificationDataset, self).__init__()

        self.categories = categories
        self.dataset = load_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation:
            augmentation = get_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        category = data["category"]

        speech, sample_rate = torchaudio.load(audio_filepath)
        for augment in self.audio_augment:
            speech = augment.apply(speech, sample_rate)

        feature = extract_filterbank(speech, sample_rate)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        category = self.categories.index(category)
        category = torch.tensor(category, dtype=torch.long)

        return feature.t(), category

    def __len__(self) -> int:
        return len(self.dataset)
