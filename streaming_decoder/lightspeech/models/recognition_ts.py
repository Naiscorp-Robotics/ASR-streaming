import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any, OrderedDict
from lightspeech.models.recognition import LightningASR, pack_input, unpack_states
from lightspeech.modules.encoder import StreamingAcousticEncoder
from lightspeech.modules.decoder import CTCDecoder
from lightspeech.datas.audio import extract_filterbank
import os

FBANK_DIM = 128          # -> extract_filterbank() default
SILENCE = "|"
FRAMERATE = 0.04
DEVICE = "cuda"
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "hop_length": 0.01,
    "segment_size": 64,
    "context_size": 16,
    "bias": 4,
    "framerate": 4
}

State = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]   # Helper Type

def load_weights(model: nn.Module, weights: OrderedDict, device: Optional[str] = "cpu"):
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

class LigtningASR_Wrapper(nn.Module):
    def __init__(self, model: nn.Module, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, feats: torch.Tensor):
        state = self.model.init_state()
        emission, length, state = self.model.stream(feats, AUDIO_CONFIG["sample_rate"], state)
        return emission, length, state

        


class LightningASR_TS(torch.nn.Module):
    def __init__(self, filepath: Any = None, model_dir: Any = None, device: Optional[str] = "cpu"):
        super().__init__()

        self.blank = 0
        self.silence = SILENCE

        # self.vocab = build_vocab()
        # self.lexicon = build_lexicon()

        self.device = device
        if not torch.jit.is_scripting() and filepath is not None and model_dir is not None: # type: ignore
            self.encoder, self.decoder = self._load_checkpoint(os.path.join(model_dir, filepath), device)

    @torch.jit.unused
    def _load_checkpoint(self, filepath: str, device: str):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]

        encoder_hparams = hparams["encoder"]
        del encoder_hparams['_target_']
        encoder = StreamingAcousticEncoder(**encoder_hparams)
        load_weights(encoder, weights["encoder"], device)

        decoder_hparams = hparams["decoder"]
        del decoder_hparams['_target_']
        decoder = CTCDecoder(**decoder_hparams)
        load_weights(decoder, weights["decoder"], device)

        return encoder, decoder

    @torch.jit.export
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

    @torch.jit.export
    def init_state(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        states = torch.jit.Attribute([], List[State])
        for _ in range(20):
            empty_memory = torch.zeros(0, 1, 512, device=self.device)
            left_context_key = torch.zeros(32, 1, 512, device=self.device)
            left_context_val = torch.zeros(32, 1, 512, device=self.device)
            past_length = torch.zeros(1, 1, dtype=torch.int32, device=self.device)
            states.append(
                (empty_memory, left_context_key, left_context_val, past_length)
            )
        return states   # (20, 4, ...)

class ModulePrototype(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.states = torch.jit.Attribute([], List[State])

    def forward(self, x: torch.Tensor) -> List[State]:
        self.states.append((x,x,x,x))
        return self.states
        

if __name__ == "__main__":
    # model = LightningASR_TS("asr-online.ckpt", "./pretrained_v2/AM", DEVICE)
    # model.eval()
    # dur = 5
    # dummy_feats = torch.randn(1, int(AUDIO_CONFIG["sample_rate"] * dur), dtype=torch.float32)
    # wrapper = LigtningASR_Wrapper(model, DEVICE)
    # wrapper.eval()
    # # traced = torch.jit.trace(wrapper, dummy_feats)
    # # torch.jit.save(traced, "./lightspeech/models/lightning_asr.ts")

    # scripted = torch.jit.script(wrapper)
    # torch.jit.save(scripted, "./lightspeech/models/lightning_asr.ts")

    model = ModulePrototype().to(DEVICE)
    model.eval()
    x = torch.randn(10, 1, 5).to(DEVICE)
    traced = torch.jit.script(model)
    torch.jit.save(traced, "./lightspeech/models/module_prototype.pt")