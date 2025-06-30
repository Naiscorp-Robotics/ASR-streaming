from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as F_audio

from lightspeech.utils.common import make_padding_mask


class AdditiveMarginSoftmaxLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        margin: float = 0.2,
        scale: float = 30,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale

        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, xs: torch.Tensor, labels: torch.LongTensor):

        x_norm = F.normalize(xs, p=2.0, dim=1)
        w_norm = F.normalize(self.W, p=2.0, dim=1)

        costh = torch.mm(x_norm, w_norm)
        delt_costh = costh.new_zeros(costh.size()).scatter_(
            1, labels.view(-1, 1), self.margin
        )

        costh_m = costh - delt_costh
        costh_m_s = self.scale * costh_m

        loss = self.cross_entropy_loss(costh_m_s, labels)
        preds = costh_m_s.softmax(dim=1).argmax(dim=1)

        return loss, preds


class RandomQuantizationLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        feature_dim: int,
        encoder_dim: int,
        quantizer_size: int,
        vocabulary_size: int,
    ):
        super(RandomQuantizationLoss, self).__init__()

        self.am_softmax_loss = AdditiveMarginSoftmaxLoss(
            encoder_dim,
            vocabulary_size,
        )

        projection = torch.randn(quantizer_size, feature_dim)
        nn.init.xavier_normal_(projection)
        self.register_buffer("projection", projection)

        codebook = torch.randn(vocabulary_size, quantizer_size)
        nn.init.normal_(codebook)
        self.register_buffer("codebook", codebook)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor, ys: torch.Tensor
    ) -> torch.Tensor:

        quantizer = F.linear(ys, self.projection)
        quantizer = F.normalize(quantizer, p=2, dim=2)[:, :, None, :]
        codebook = F.normalize(self.codebook, p=2, dim=1)[None, None, :, :]
        targets = F.pairwise_distance(quantizer, codebook).argmin(dim=2)

        masks = make_padding_mask(x_lens, xs.size(1))
        loss, __ = self.am_softmax_loss(xs[masks], targets[masks])

        return loss


class SequenceToSequenceLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        ctc_weight: float = 1.0,
        att_weight: float = 1.0,
    ):
        super(SequenceToSequenceLoss, self).__init__()
        self.blank_label = 0
        self.ctc_weight = ctc_weight
        self.att_weight = att_weight

    def forward(
        self,
        ctc_logits: torch.Tensor,
        rnnt_logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:

        ctc_loss = F.ctc_loss(
            log_probs=ctc_logits.transpose(0, 1),
            targets=targets,
            input_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=self.blank_label,
            zero_infinity=True,
        )

        rnnt_loss = F_audio.rnnt_loss(
            logits=rnnt_logits,
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.blank_label,
        )

        loss = self.ctc_weight * ctc_loss + self.rnnt_weight * rnnt_loss

        return loss, ctc_loss, rnnt_loss


class LeastSquaresGenerativeLoss(nn.modules.loss._Loss):
    def forward(self, disc_outs: List[torch.Tensor]) -> torch.Tensor:

        loss = 0.0
        for dg in disc_outs:
            loss += torch.mean((1 - dg) ** 2)

        loss = loss / len(disc_outs)

        return loss


class LeastSquaresAdversarialLoss(nn.modules.loss._Loss):
    def forward(
        self,
        disc_outs: List[torch.Tensor],
        disc_tgts: List[torch.Tensor],
    ) -> torch.Tensor:

        loss = 0.0
        for dg, dr in zip(disc_outs, disc_tgts):
            loss += torch.mean((1 - dr) ** 2) + torch.mean(dg**2)

        loss = loss / len(disc_tgts)

        return loss


class STFTLoss(nn.modules.loss._Loss):
    def __init__(self, n_fft: int, win_length: int, hop_length: int):
        super(STFTLoss, self).__init__()
        self.spectrogram = T.Spectrogram(
            n_fft,
            win_length,
            hop_length,
            power=1,
        )

    def forward(
        self,
        audio_outs: torch.Tensor,
        audio_tgts: torch.Tensor,
        audio_masks: torch.Tensor,
    ) -> torch.Tensor:

        spec_outs = self.spectrogram(audio_outs)
        spec_tgts = self.spectrogram(audio_tgts)

        masks = F.interpolate(audio_masks.float(), spec_tgts.size(2))
        masks = masks.bool().expand_as(spec_tgts)

        sc_loss = self.spectral_convergence_loss(spec_outs, spec_tgts, masks)
        mag_loss = self.log_stft_magnitude_loss(spec_outs, spec_tgts, masks)

        loss = sc_loss + mag_loss

        return loss

    def spectral_convergence_loss(
        self, xs: torch.Tensor, ys: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:

        numerator = ((ys - xs) * masks).norm(p="fro")
        denominator = (ys * masks).norm(p="fro")

        loss = numerator / (denominator + 1e-9)

        return loss

    def log_stft_magnitude_loss(
        self, xs: torch.Tensor, ys: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:

        xs = xs.add(1e-9).log()
        ys = ys.add(1e-9).log()

        loss = F.l1_loss(xs, ys, reduction="none")
        loss = (loss * masks).sum() / masks.sum()

        return loss


class MultiResolutionSTFTLoss(nn.modules.loss._Loss):
    def __init__(self, resolutions: List[Tuple[int, int, int]]):
        super(MultiResolutionSTFTLoss, self).__init__()

        self.stft_losses = nn.ModuleList()
        for fs, wl, hl in resolutions:
            self.stft_losses.append(STFTLoss(fs, wl, hl))

    def forward(
        self,
        audio_outs: torch.Tensor,
        audio_tgts: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:

        audio_outs = audio_outs.squeeze(1)
        audio_tgts = audio_tgts.squeeze(1)

        audio_masks = make_padding_mask(audio_lens, audio_outs.size(1))
        audio_masks = audio_masks[:, None, :]

        loss = 0.0
        for stft in self.stft_losses:
            loss += stft(audio_outs, audio_tgts, audio_masks)

        loss = loss / len(self.stft_losses)

        return loss


class TemporalPredictionLoss(nn.modules.loss._Loss):
    def __init__(self, min_value=-100.0):
        super(TemporalPredictionLoss, self).__init__()
        self.min_value = min_value

    def forward(self, outs: torch.Tensor, tgts: torch.Tensor) -> torch.Tensor:
        outs = outs.log().clamp(self.min_value)
        tgts = tgts.log().clamp(self.min_value)

        masks = tgts != self.min_value

        loss = F.mse_loss(outs, tgts, reduction="none")
        loss = (loss * masks).sum() / masks.sum()

        return loss
