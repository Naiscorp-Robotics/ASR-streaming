import os
import re
import tarfile
import tempfile
from typing import List

from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lightspeech.datas.text import (
    build_dictionary,
    build_vocabulary,
    tokenize_subword,
)
from lightspeech.utils.common import load_module


def load_checkpoint(filepath: str, device: str):

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(filepath, "r") as tar:
            tar.extractall(path=tmpdir)

        config = OmegaConf.load(os.path.join(tmpdir, "config.yaml"))

        cfg = config.model.encoder
        path = os.path.join(tmpdir, "encoder.pt")
        encoder = load_module(cfg, path, device)

        cfg = config.model.decoder
        path = os.path.join(tmpdir, "decoder.pt")
        decoder = load_module(cfg, path, device)

    return encoder, decoder


# class LightningTTS(nn.Module):
#     def __init__(self, filepath: str, device: str = "cpu"):
#         super(LightningTTS, self).__init__()

#         self.blank = 0
#         self.device = device

#         self.dictionary = build_dictionary()
#         self.vocabulary = build_vocabulary()
#         self.encoder, self.decoder = load_checkpoint(filepath, device)

#     def forward(
#         self,
#         texts: List[str],
#         style_embs: torch.Tensor = None,
#         time_scale: float = 1.0,
#     ) -> List[torch.Tensor]:

#         batch = [self._tokenize(sent) for sent in texts]

#         xs = [b[0] for b in batch]
#         x_lens = [len(x) for x in xs]
#         word_idxs = [b[1] for b in batch]

#         xs = pad_sequence(xs, batch_first=True)
#         x_lens = torch.tensor(x_lens, device=self.device)
#         word_idxs = pad_sequence(word_idxs, batch_first=True)

#         enc_outs, enc_lens, __ = self.encoder(
#             xs, x_lens, word_idxs, time_scale=time_scale
#         )
#         audio_outs, audio_lens = self.decoder(enc_outs, enc_lens, style_embs)

#         bs = len(texts)
#         outputs = [audio_outs[i, :, : audio_lens[i]] for i in range(bs)]

#         return outputs

#     def _tokenize(self, sentence: str) -> List[int]:

#         # Replace punctuation marks with blank token
#         words = [re.sub(r"[^\w\s<>]", "", word) for word in sentence.split()]

#         # Insert OOV tag
#         for i, word in enumerate(words):
#             if (word not in self.dictionary) and (word != ""):
#                 words[i] = "<<" + word + ">>"

#         # Insert SOS token and EOS token
#         words = [""] + words if words[0] != "" else words
#         words = words + [""] if words[-1] != "" else words

#         token_idxs, word_idxs = [], []
#         for word in words:
#             word_idx = max(word_idxs) + 1 if len(word_idxs) > 0 else 0

#             if word == "":
#                 token_idxs += [self.blank]
#                 word_idxs += [word_idx]
#             else:
#                 tokens = tokenize_subword(word, self.vocabulary)
#                 token_idxs += [self.vocabulary.index(t) + 1 for t in tokens]
#                 word_idxs += [word_idx] * len(tokens)

#         token_idxs = torch.tensor(token_idxs, device=self.device)
#         word_idxs = torch.tensor(word_idxs, device=self.device)

#         return token_idxs, word_idxs


class LightningTTS(nn.Module):
    def __init__(
        self,
        acoustic_filepath: str,
        vocoder_filepath: str,
        device: str = "cpu",
    ):
        super(LightningTTS, self).__init__()

        self.blank = 0
        self.device = device

        self.dictionary = build_dictionary()
        self.vocabulary = build_vocabulary()

        self.acoustic_encoder, self.acoustic_decoder = load_checkpoint(
            acoustic_filepath, device
        )
        self.vocoder_encoder, self.vocoder_decoder = load_checkpoint(
            vocoder_filepath, device
        )

    def forward(
        self,
        texts: List[str],
        style_embs: torch.Tensor = None,
        time_scale: float = 1.0,
    ) -> List[torch.Tensor]:

        batch = [self._tokenize(sent) for sent in texts]

        xs = [b[0] for b in batch]
        x_lens = [len(x) for x in xs]
        word_idxs = [b[1] for b in batch]

        xs = pad_sequence(xs, batch_first=True)
        x_lens = torch.tensor(x_lens, device=self.device)
        word_idxs = pad_sequence(word_idxs, batch_first=True)

        enc_outs, enc_lens, __ = self.acoustic_encoder(
            xs,
            x_lens,
            word_idxs,
            time_scale=time_scale,
        )
        mel_outs, mel_lens = self.acoustic_decoder(
            enc_outs,
            enc_lens,
            style_embs,
        )

        enc_outs, enc_lens = self.vocoder_encoder(mel_outs, mel_lens)
        audio_outs, audio_lens = self.vocoder_decoder(enc_outs, enc_lens)

        bs = len(texts)
        audios = [audio_outs[i, :, : audio_lens[i]] for i in range(bs)]

        return audios

    def _tokenize(self, sentence: str) -> List[int]:

        # Replace punctuation marks with blank token
        words = [re.sub(r"[^\w\s<>]", "", word) for word in sentence.split()]

        # Insert OOV tag
        for i, word in enumerate(words):
            if (word not in self.dictionary) and (word != ""):
                words[i] = "<<" + word + ">>"

        # Insert SOS token and EOS token
        words = [""] + words if words[0] != "" else words
        words = words + [""] if words[-1] != "" else words

        token_idxs, word_idxs = [], []
        for word in words:
            word_idx = max(word_idxs) + 1 if len(word_idxs) > 0 else 0

            if word == "":
                token_idxs += [self.blank]
                word_idxs += [word_idx]
            else:
                tokens = tokenize_subword(word, self.vocabulary)
                token_idxs += [self.vocabulary.index(t) + 1 for t in tokens]
                word_idxs += [word_idx] * len(tokens)

        token_idxs = torch.tensor(token_idxs, device=self.device)
        word_idxs = torch.tensor(word_idxs, device=self.device)

        return token_idxs, word_idxs
