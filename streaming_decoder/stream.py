import torch
import time
from omegaconf import OmegaConf
from utils import AudioConfig, compute_relative_cost, logger
import wave
import webrtcvad

from online_endpoint import load_endpointing_rule, detect_endpointing

class Stream(AudioConfig):
    def __init__(self, config: OmegaConf) -> None:
        """
        config: audio config
        """
        if config.language == "vi":
            AudioConfig.__init__(self, config.audio)
        else:
            AudioConfig.__init__(self, config.audio_en)

        self.frame_shift_ms          = int(self.hop_length/self.sample_rate*1000)
        self.language                = config.language
        # streaming data
        self.audio_stream            = torch.zeros(self.buffer_length)
        self.audio_total             = torch.Tensor([])
        self.audio_total_len         = 0
        self.length_of_segment       = self.buffer_length
        self.chunk_processed         = 0
        self.num_audio_accept        = 0
        self.chunk_processed_total   = 0
        self.trailing_blank_duration = 0

        self.sw_model                = "GENERAL"
        self.is_sw_model             = False
        self.id                      = ""
        self.is_eos                  = False
        self.offset                  = - (self.context_size // self.framerate + 1)
        self.input_sr                = self.sample_rate
        self.state                   = None
        self.hypothesis              = None
        self.transcript_internal     = ""
        self.transcript              = ""
        self.is_contain_token        = False
        self.is_new_segment          = True
        self.offset_compute_stats    = 0.0
        self.segment_start           = 0.0
        self.segment_end             = 0.0
        # init LM
        self.searcher                = None
        self.emission                = torch.Tensor([])
        self.length                  = 0

        # VAD
        self.vad_conf               = config.Vad
        self.VADWebrtc_Computer     = webrtcvad.Vad(self.vad_conf.Webrtc.aggressiveness)
        self.vadwebrtc_chunk_length = int(self.vad_conf.Webrtc.chunk_duration * self.sample_rate) * 2

        # whenever an endpoint is detected, it is incremented
        self.segment    = 0

        # Endpointing
        self.mapping_endpointing_rule = config.Mapping_rule
        self.EndpointingRule = {}
        for rule_type in config.Endpointing_rules.keys():
            self.EndpointingRule[rule_type] = load_endpointing_rule(
                config.Endpointing_rules[rule_type]
            )

    def init_save_audio(self, audio_path):
        self.audio_file = wave.open(audio_path, "wb")
        self.audio_file.setnchannels(1)  # Mono
        self.audio_file.setsampwidth(2)  # 16-bit samples
        self.audio_file.setframerate(self.sample_rate)

    def done_save_audio(self):
        self.audio_file.setnframes(self.audio_total_len)
        self.audio_file.close()

    def accept_waveform(
            self,
            waveform: torch.Tensor,
    ) -> None:
        if waveform.size(0) > 100:
            self.audio_stream       = torch.cat((self.audio_stream, waveform), dim=0)
            self.length_of_segment += waveform.size(0)
            self.num_audio_accept  += 1
            self.audio_total        = torch.cat((self.audio_total, waveform), dim=0)
            self.audio_total_len   += waveform.size(0) * 2

    def discard_decoded_segment(self, segment_length, save_audio):
        if save_audio:
            segment = self.audio_total[:int(segment_length*self.sample_rate)]
            self.audio_file.writeframes(segment.numpy().tobytes())
        self.audio_total = self.audio_total[int(segment_length*self.sample_rate):]
        self.offset_compute_stats += segment_length

    def add_tail_paddings(self) -> None:
        """Add some tail paddings so that we have enough context to process
        segment at the very end of an utterance.

        Args:
            n:
                Number of tail padding samples to be added. You can increase it if
                it happens that there are many missing tokens for the last word of
                an utterance.
        """
        n = self.chunk_length - self.audio_stream.size(0)
        tail_padding      = torch.zeros(n)
        self.audio_stream = torch.cat((self.audio_stream, tail_padding), dim = 0)

    def update_stream(self, text, last_blank):
        if self.emission.size(0) == 16:
            self.offset     = int(self.chunk_processed_total * self.segment_size / self.bias) - \
                (self.context_size // self.framerate + 1)
        if self.language == "vi":
            self.transcript_internal    = text
        else:
            self.transcript_internal += text
        self.chunk_processed       += 1
        self.chunk_processed_total += 1

        if text:
            self.trailing_blank_duration  = last_blank
            self.is_contain_token         = True
        else:
            self.trailing_blank_duration += 0.64 if self.language == "vi" else 0.16

    def endpoint_detected(self, ngram, prob) -> (bool, float):
        """
        Returns:
            Return True if endpoint is detected; return False otherwise.
        """
        length_utt_decoded      = (
            self.chunk_processed * self.segment_length
        ) / self.sample_rate

        relative_cost = compute_relative_cost(self.transcript_internal, ngram, prob)
        logger.debug(f"[{self.id}] transcript_internal: {self.transcript_internal}, {relative_cost}")

        endpointing_rule        = self.mapping_endpointing_rule[self.sw_model]
        self.trailing_blank_duration = round(self.trailing_blank_duration, 2)
        detected, rule_activate, _   = detect_endpointing(
                            rule=self.EndpointingRule[endpointing_rule],
                            utterance_length=length_utt_decoded,
                            trailing_silence=self.trailing_blank_duration,
                            relative_cost=relative_cost
            )
        logger.debug(f"[{self.id}] trailing_silence: {self.trailing_blank_duration}, {length_utt_decoded}")
        if detected:
            self.segment_end = self.trailing_blank_duration
            self.transcript = self.transcript_internal
            logger.info(f"[{self.id}] Type endpointing: {endpointing_rule}, rule activate: {rule_activate}, segment_length: {length_utt_decoded}, trailing_silence: {self.trailing_blank_duration}, relative_cost: {relative_cost}")
            self.chunk_processed         = 0
            self.is_contain_token        = False
            self.trailing_blank_duration = 0
            self.segment                += 1
            self.transcript_internal     = ""

        # Update stream audio
        self.audio_stream       = self.audio_stream[self.segment_length:]
        self.length_of_segment -= self.segment_length


        return detected, length_utt_decoded


    def detect_speech(self):
        chunk_audio = self.audio_stream[self.buffer_length:self.chunk_length]
        # State 1: Vad webrtc
        chunk_audio_bytes = (chunk_audio * 32768.0).to(torch.int16).numpy().tobytes()
        num_frame = len(chunk_audio_bytes) // self.vadwebrtc_chunk_length
        contain_speech = False
        start_time = time.time()
        for i in range(num_frame):
            Frame = chunk_audio_bytes[i*self.vadwebrtc_chunk_length: (i+1)*self.vadwebrtc_chunk_length]
            is_speech = self.VADWebrtc_Computer.is_speech(Frame, self.sample_rate)
            if is_speech:
                contain_speech = True
                break

        logger.debug(f"[{self.id}] vad webrtc: {self.chunk_processed} {contain_speech} {time.time() - start_time}")

        if not contain_speech:
            self.trailing_blank_duration += 0.64 if self.language == "vi" else 0.16
            self.chunk_processed         += 1
            self.chunk_processed_total   += 1
            if self.emission.size(0) != 0:
                self.offset              += int(self.segment_size / self.bias)

        return contain_speech