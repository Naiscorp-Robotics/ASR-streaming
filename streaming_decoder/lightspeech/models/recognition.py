import re, os
import math
from typing import Tuple, List, Dict, Optional

import torch, torchaudio
from torchaudio.models.decoder import ctc_decoder, CTCHypothesis
from torchaudio.pipelines.rnnt_pipeline import (
    _ModuleFeatureExtractor, 
    _FunctionalModule, 
    _GlobalStatsNormalization,
    _SentencePieceTokenProcessor,
    _piecewise_linear_log,
    _gain
)
from torchaudio.models.rnnt_decoder import Hypothesis
from torchaudio.models import emformer_rnnt_base, RNNTBeamSearch


from lightspeech.datas.audio import extract_filterbank
from lightspeech.datas.text import build_vocab, build_lexicon, tokenize
from lightspeech.utils.common import load_module
from lightspeech.utils.alignment import (
    get_trellis,
    backtrack,
    merge_tokens,
    merge_words,
)

SILENCE = "|"
FRAMERATE = 0.04

vocab = build_vocab()
def greedy_search(
    emission: None
) -> Tuple[str, float]:

    indices = torch.argmax(emission, dim=1)

    last_blank = FRAMERATE * len(emission)
    tokens_idx = (indices > 1).nonzero(as_tuple=True)[0]
    if len(tokens_idx):
        last_blank = (len(indices) - 1 - tokens_idx[-1]) * FRAMERATE
        last_blank = last_blank.item()
    indices = torch.unique_consecutive(indices, dim=0)
    indices = torch.masked_select(indices, indices != 0)

    tokens = [vocab[idx] for idx in indices if idx != 0]

    text = "".join(tokens)
    text = text.replace("<<", "").replace(">>", "")
    text = text.replace("-", "").replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()

    score = torch.amax(emission, dim=1).sum() / indices.size(0)
    score = score.exp().item()

    return text, last_blank


def pack_input(states: List):
    state = []
    batch_size = len(states)
    for layer in range(20):
        list_memory = [states[idx][layer][0] for idx in range(batch_size)]
        list_key    = [states[idx][layer][1] for idx in range(batch_size)]
        list_val    = [states[idx][layer][2] for idx in range(batch_size)]
        list_length = [states[idx][layer][3] for idx in range(batch_size)]

        memory           = torch.cat(list_memory, dim=1)
        left_context_key = torch.cat(list_key, dim=1)
        left_context_val = torch.cat(list_val, dim=1)
        past_length      = torch.cat(list_length, dim=1)
        state.append([memory, left_context_key, left_context_val, past_length])

    return state

def unpack_states(states: List):
    batch_size = states[0][0].size(1)

    batch_state = []
    for idx in range(batch_size):
        layer_state = []
        for layer in range(20):
            memory = states[layer][0][:, idx, :].unsqueeze(1)
            left_context_key = states[layer][1][:, idx, :].unsqueeze(1)
            left_context_val = states[layer][2][:, idx, :].unsqueeze(1)
            past_length = states[layer][3][:, idx].unsqueeze(1)
            layer_state.append(
                [memory, left_context_key, left_context_val, past_length]
            )
        batch_state.append(layer_state)
    return batch_state



class EmformerRNNT:
    def __init__(self, model_dir: str, device: Optional[str] = "cpu"):
        super(EmformerRNNT, self).__init__()
        model_dir = os.path.dirname(model_dir)
        self.feature_extractor = _ModuleFeatureExtractor(
            torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_fft=400, n_mels=80, hop_length=160
                ),
                _FunctionalModule(lambda x: x.transpose(1, 0)),
                _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
                _GlobalStatsNormalization(f"{model_dir}/emformer-rnnt/global_stats_rnnt.json"),
            )
        )

        # Model
        model = emformer_rnnt_base(num_symbols=4097)
        state_dict = torch.load(f"{model_dir}/emformer-rnnt/emformer_rnnt.pt")
        model.load_state_dict(state_dict)
        model.eval()
        self.decoder = RNNTBeamSearch(model, blank=4096)

        # token_processor
        self.token_processor = _SentencePieceTokenProcessor(f"{model_dir}/emformer-rnnt/spm_bpe_4096.model")


    @torch.inference_mode()
    def stream(
        self,
        speech: torch.Tensor,
        state: List[List[torch.Tensor]],
        hypothesis: Optional[Hypothesis],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        features, length = self.feature_extractor(speech)
        import logging
        logging.info(f"{features.size()}, {length}")
        hypos, state = self.decoder.infer(features, length, beam_width=10, state=state, hypothesis=hypothesis)
        return hypos[0], state


class LightningASR:
    def __init__(self, filepath: str, model_dir: str, device: Optional[str] = "cpu"):
        super(LightningASR, self).__init__()

        self.blank = 0
        self.silence = SILENCE

        self.vocab = build_vocab()
        self.lexicon = build_lexicon()

        self.device = device
        self.encoder, self.decoder = self._load_checkpoint(os.path.join(model_dir, filepath), device)

    def _load_checkpoint(self, filepath: str, device: str):

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]

        encoder = load_module(hparams["encoder"], weights["encoder"], device)
        decoder = load_module(hparams["decoder"], weights["decoder"], device)

        return encoder, decoder


    def force_alignment(
        self,
        speeches: List[torch.Tensor],
        sample_rate: int,
        transcripts: List[str],
    ):

        emissions, lengths = self(speeches, sample_rate)

        alignments = []
        for i, emission in enumerate(emissions):
            emission = emission[: lengths[i]]

            length   = lengths[i].item()
            duration = speeches[i].size(1) / sample_rate

            tokens   = tokenize(transcripts[i], self.vocab, self.lexicon)
            token_indices = [self.vocab.index(token) for token in tokens]

            trellis = get_trellis(emission, token_indices, self.blank)
            path    = backtrack(trellis, emission, token_indices, self.blank)

            token_segments = merge_tokens(path, tokens, length, duration)
            word_segments  = merge_words(token_segments, self.silence)

            alignments.append((token_segments, word_segments))

        return alignments

    @torch.inference_mode()
    def stream(
        self,
        speeches: torch.Tensor,
        sample_rate: int,
        states: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        states = pack_input(states)
        xs, x_lens = extract_filterbank(torch.cat(speeches).to(self.device), sample_rate, self.device)

        enc_outs, enc_lens, states = self.encoder.infer(xs, x_lens, states)
        states   = unpack_states(states)
        dec_outs = self.decoder(enc_outs)
        return dec_outs.cpu(), enc_lens.cpu(), states


    def init_state(self) -> List[torch.Tensor]:
        states = []
        for layer in range(20):
            empty_memory = torch.zeros(0, 1, 512, device=self.device)
            left_context_key = torch.zeros(32, 1, 512, device=self.device)
            left_context_val = torch.zeros(32, 1, 512, device=self.device)
            past_length = torch.zeros(1, 1, dtype=torch.int32, device=self.device)
            states.append(
                [empty_memory, left_context_key, left_context_val, past_length]
            )
        return states


class BeamSearchDecoder:
    def __init__(
        self,
        lexicon: str,
        lm: str,
        corpus_dir: str,
        lm_weight: float = 1.0,
        beam_size: int = 50,
        beam_size_token: int = 5,
        beam_threshold: float = 50.0,
        word_score: float = 0.5,
    ):
        self.silence = SILENCE
        self.framerate = FRAMERATE

        self.vocab = build_vocab()
        self.decoder = ctc_decoder(
            tokens=self.vocab,
            lexicon=os.path.join(corpus_dir, lexicon),
            lm=os.path.join(corpus_dir, lm),
            lm_weight=lm_weight,
            beam_size=beam_size,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            word_score=word_score,
        )

    def transcript_offline(
        self, emission: torch.Tensor, length: torch.Tensor, offset: int
    ) -> List[Dict]:

        emission = emission[: length]
        T, N = emission.size()

        result = self.decoder.decoder.decode(emission.data_ptr(), T, N)[0]
        hypo = CTCHypothesis(
            tokens=self.decoder._get_tokens(result.tokens),
            words=[word for word in result.words if word >= 0],
            score=result.score,
            timesteps=self.decoder._get_timesteps(result.tokens),
        )

        transcript, __, alignment = self._analyze_hypothesis(hypo, offset=offset)

        return alignment


    def _analyze_hypothesis(self, hypothesis: CTCHypothesis, offset: int):

        tokens = self.decoder.idxs_to_tokens(hypothesis.tokens)
        timesteps = hypothesis.timesteps.tolist()

        word_indices = hypothesis.words
        words = [self.decoder.word_dict.get_entry(idx) for idx in word_indices]

        transcript = " ".join(words)
        score = math.exp(hypothesis.score / (len(hypothesis.tokens) + 1))
        alignment = []
        item = {"beg": 0, "end": 0, "word": [], "confidence": 0.0}

        for i in range(len(tokens)):
            if (i == 0 and tokens[i] != self.silence) or (
                i != 0 and tokens[i - 1] == self.silence
            ):
                timestep = (timesteps[i] + offset) * self.framerate
                item["beg"] = round(timestep, 2)

            if tokens[i] != self.silence:
                item["word"].append(tokens[i])
            elif i != 0:
                timestep = (timesteps[i] + offset) * self.framerate
                item["end"] = round(timestep, 2)
                item["word"] = "".join(item["word"])
                item["confidence"] = round(score, 2)

                alignment.append(item)
                item = {"beg": 0, "end": 0, "word": [], "confidence": 0.0}

        alignment = [align for align in alignment if align["word"] != ""]

        return transcript, score, alignment

