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
        
        import logging
        logging.info(f"Initializing EmformerRNNT with device: {device}")
        
        # Đảm bảo device là cuda nếu có thể
        # Kiểm tra CUDA availability trước mọi thứ
        cuda_available = torch.cuda.is_available()
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Kiểm tra bộ nhớ GPUe()
        logging.info(f"CUDA available check: {cuda_available}")
        
        if cuda_available:
            # Buộc sử dụng CUDA
            device = "cuda"
            torch.cuda.set_device(0)  # Đặt thiết bị CUDA về 0
            logging.info(f"CUDA is available with {torch.cuda.device_count()} devices")
            logging.info(f"Initial GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
        else:
            logging.warning("CUDA is NOT available, falling back to CPU")
            device = "cpu"
            
        # Đảm bảo device là torch.device
        self.device = torch.device(device)
        logging.info(f"Final device selection: {self.device}")
        
        # Khởi tạo feature extractor - đặt lên device
        fe_components = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=400, n_mels=80, hop_length=160
            ),
            _FunctionalModule(lambda x: x.transpose(1, 0)),
            _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
            _GlobalStatsNormalization(f"{model_dir}/emformer-rnnt/global_stats_rnnt.json"),
        ).to(self.device)
        
        self.feature_extractor = _ModuleFeatureExtractor(fe_components)
        logging.info(f"Feature extractor created and moved to {self.device}")

        # Khởi tạo model RNNT trực tiếp trên device đã chọn
        try:
            # Tải state dict vào đúng device
            logging.info(f"Loading model state dict to {self.device}")
            state_dict = torch.load(f"{model_dir}/emformer-rnnt/emformer_rnnt.pt", map_location=self.device)
            
            # Khởi tạo model
            model = emformer_rnnt_base(num_symbols=4097)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Di chuyển model lên device
            model = model.to(self.device)
            logging.info(f"Model successfully moved to {self.device}")
            
            # Test bằng cách thực hiện forward một tensor nhỏ
            if self.device.type == "cuda":
                try:
                    # Không truy cập encoder trực tiếp vì mô hình có thể không có thuộc tính này
                    # Thay vào đó, thực hiện một phép tính đơn giản trên GPU để kiểm tra
                    test_tensor = torch.zeros(1, device=self.device)
                    test_tensor = test_tensor + 1
                    torch.cuda.synchronize()
                    logging.info("Successfully performed a simple CUDA operation")
                    
                    # Kiểm tra xem tensor có thực sự ở trên GPU không
                    logging.info(f"Test tensor device: {test_tensor.device}")
                except Exception as e:
                    logging.error(f"Error in CUDA test: {e}")
                    
            # Kiểm tra vị trí của các tham số
            cuda_params = 0
            cpu_params = 0
            for name, param in model.named_parameters():
                if param.device.type == "cuda":
                    cuda_params += 1
                else:
                    cpu_params += 1
                    
            logging.info(f"Model parameters on CUDA: {cuda_params}, on CPU: {cpu_params}")
            
        except Exception as e:
            logging.error(f"Error setting up model: {e}")
            # Không chuyển sang CPU khi có lỗi, vẫn cố gắng sử dụng GPU
            if self.device.type == "cuda":
                logging.warning("Error occurred but still trying to use CUDA")
                # Khởi tạo model một cách đơn giản
                model = emformer_rnnt_base(num_symbols=4097)
                state_dict = torch.load(f"{model_dir}/emformer-rnnt/emformer_rnnt.pt", map_location=self.device)
                model.load_state_dict(state_dict)
                model.eval()
                # Đảm bảo mô hình nằm trên GPU
                model = model.to(self.device)
                logging.info(f"Model reinitialized on {self.device}")
                torch.cuda.empty_cache()  # Dọn dẹp bộ nhớ GPU
        
        # Thiết lập decoder với model đã được di chuyển lên device
        self.decoder = RNNTBeamSearch(model, blank=4096)
        
        # Khởi tạo bộ xử lý token
        self.token_processor = _SentencePieceTokenProcessor(f"{model_dir}/emformer-rnnt/spm_bpe_4096.model")
        
        # Ghi log bộ nhớ cuối cùng
        if self.device.type == "cuda":
            logging.info(f"Final GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")


    @torch.inference_mode()
    def stream(
        self,
        speech: torch.Tensor,
        state: List[List[torch.Tensor]],
        hypothesis: Optional[Hypothesis],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        import logging
        
        try:
            # Khởi tạo CUDA nếu cần
            if self.device.type == "cuda" and torch.cuda.is_available():
                # Đảm bảo CUDA được kích hoạt
                torch.cuda.synchronize()
                logging.info(f"CUDA memory at start of stream: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
                
            # Đảm bảo tensor dữ liệu vào được chuyển tới GPU
            speech = speech.to(self.device, non_blocking=True)  # non_blocking=True để tăng tốc chuyển dữ liệu
            
            # Chuẩn bị state nếu cần
            if state:
                try:
                    state = [[tensor.to(self.device, non_blocking=True) 
                            if isinstance(tensor, torch.Tensor) else tensor     
                            for tensor in layer] 
                            for layer in state]
                except Exception as e:
                    logging.error(f"Error moving state to device: {e}")
            
            # Thực hiện trích xuất đặc trưng
            features, length = self.feature_extractor(speech)
            
            # Đảm bảo dữ liệu đặc trưng ở trên GPU
            features = features.to(self.device)
            length = length.to(device=self.device)
            
            # Ghi nhật ký trạng thái hiện tại
            logging.info(f"Features shape: {features.shape}, on device: {features.device}")
            
            # Thực hiện inference
            if self.device.type == "cuda":
                # Ghi log thông tin bộ nhớ trước khi inference
                logging.info(f"GPU memory before inference: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
                
                # Đảm bảo CUDA không bị memory fragmentation
                torch.cuda.empty_cache()
            
            # Thực hiện inference - luôn sử dụng đúng device
            hypos, state = self.decoder.infer(features, length, beam_width=10, state=state, hypothesis=hypothesis)
            
            if self.device.type == "cuda":
                # Đồng bộ để đảm bảo tính toán hoàn tất trên GPU
                torch.cuda.synchronize()
                
                # Ghi log thông tin bộ nhớ sau khi inference
                logging.info(f"GPU memory after inference: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
            
            # Fix: Check if hypos is empty before accessing index 0
            if not hypos:
                logging.warning("No hypotheses returned from decoder - returning empty hypothesis")
                if hypothesis is None:
                    empty_hypo = Hypothesis(tokens=[], token_ids=[], score=0.0)
                else:
                    empty_hypo = hypothesis
                return empty_hypo, state
            
            return hypos[0], state
            
        except Exception as e:
            logging.error(f"Error in stream method: {e}")
            if hypothesis is None:
                empty_hypo = Hypothesis(tokens=[], token_ids=[], score=0.0)
            else:
                empty_hypo = hypothesis
            return empty_hypo, state


class LightningASR:
    def __init__(self, filepath: str, model_dir: str, device: Optional[str] = "cpu"):
        super(LightningASR, self).__init__()
        
        # Add logging for diagnostic
        import logging
        logging.info(f"Initializing LightningASR with device: {device}")

        self.blank = 0
        self.silence = SILENCE

        self.vocab = build_vocab()
        self.lexicon = build_lexicon()

        # Store device for later use
        self.device = device
        
        # Check device setup
        if device == "cuda" and torch.cuda.is_available():
            logging.info(f"CUDA is available for LightningASR")
            logging.info(f"Memory allocated before model load: {torch.cuda.memory_allocated(0) / 1024**2} MB")
        
        # Load models to the specified device
        self.encoder, self.decoder = self._load_checkpoint(os.path.join(model_dir, filepath), device)
        
        if device == "cuda" and torch.cuda.is_available():
            logging.info(f"Memory allocated after model load: {torch.cuda.memory_allocated(0) / 1024**2} MB")
            
    def _load_checkpoint(self, filepath: str, device: str):
        import logging
        logging.info(f"Loading checkpoint from {filepath} to device {device}")

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]

        # Load modules explicitly to the target device
        encoder = load_module(hparams["encoder"], weights["encoder"], device)
        decoder = load_module(hparams["decoder"], weights["decoder"], device)
        
        logging.info(f"Loaded encoder and decoder to {device}")

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

