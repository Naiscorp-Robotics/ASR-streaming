import time
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import numpy as np

from stream import Stream
from utils import AudioConfig, logger
from lightspeech.models.recognition import greedy_search

from threading import current_thread
from vad_silero import get_speech_probs, OnnxWrapper


class StreamingE2E(AudioConfig):
    def __init__(self, config: DictConfig, device="cpu"):
        AudioConfig.__init__(self, config.audio)


        # model
        self.model           = instantiate(config.Acoustic_Model, model_dir=config.model_dir, device=device)
        self.state_init      = self.model.init_state()
        self.VADSilero_model = OnnxWrapper(config.Vad.Silero.model_path, config.Vad.Silero.force_onnx_cpu)
        # init searcher

    def init_stream(self, stream: Stream, init_searcher: True) -> None:
        """
        Initiating context
        Args:
            stream: Stream
        """
        if stream.state == None:
            stream.state       = self.state_init
        if init_searcher:
            logger.info("Init searcher!")
            stream.offset      = - (self.context_size // self.framerate + 1)


    @torch.no_grad()
    def process(
        self,
        stream_list,
    ):
        thread = current_thread()

        # Stage 1: Batching request
        try:
            speeches_batch, states_batch = [], []
            speeches_silero_batch = []
            index_silero, index_asr = [], []
            for i, stream in enumerate(stream_list):
                chunk = stream.audio_stream[:self.chunk_length]
                if not stream.is_contain_token:
                    index_silero.append(i)
                    stream.n_silero += 1
                    speeches_silero_batch.append(chunk.unsqueeze(0))
                else:
                    index_asr.append(i)
                    speeches_batch.append(chunk.unsqueeze(0))
                    states_batch.append(stream.state)

        except Exception as inst:
            logger.error(f"Error when batching request: {inst}")

        # State 2: Vad Silero
        if len(speeches_silero_batch) != 0:
            try:
                silero_batch = torch.concat(speeches_silero_batch, dim=0)#[:, self.buffer_length:]
                start_time = time.time()
                speeches, speech_probs = get_speech_probs(silero_batch, self.VADSilero_model, sampling_rate=self.sample_rate)
                index_speech, index_nonspeech = np.where(speeches)[0], np.where(~speeches)[0]
                logger.debug(f"vad silero: {silero_batch.size(0)} {speech_probs} {time.time() - start_time}")
                logger.debug(f"index_speech: {index_speech}, index_nonspeech: {index_nonspeech}")
                if len(index_speech) != 0:
                    for i in index_speech:
                        index_asr.append(index_silero[i])
                        speeches_batch.append(speeches_silero_batch[i])
                        states_batch.append(stream_list[index_silero[i]].state)
                if len(index_nonspeech) != 0:
                    for i in index_nonspeech:
                        stream_list[index_silero[i]].trailing_blank_duration += 0.64
                        stream_list[index_silero[i]].chunk_processed         += 1
                        stream_list[index_silero[i]].chunk_processed_total   += 1
                        if stream_list[index_silero[i]].emission.size(0) != 0:
                            stream_list[index_silero[i]].offset              += int(stream_list[index_silero[i]].segment_size / stream_list[index_silero[i]].bias)
            except Exception as inst:
                logger.error(f"Error when vad silero: {inst}")

        # Stage 2: Decode AM
        start = time.time()
        if len(speeches_batch) != 0:
            try:
                emission, length, states_batch = self.model.stream(
                        speeches_batch, self.sample_rate, states_batch
                )
                logger.debug(f"************ [thread-{thread.name}] Decode AM time: {time.time() - start}, batchsize: {len(speeches_batch)}")
            except Exception as inst:
                logger.error(f"Error when decode AM: {inst}")


            # Stage 3: Decode LM
            try:
                start = time.time()
                for idx, i in enumerate(index_asr):
                    stream_list[i].n_decode += 1
                    stream_list[i].state = states_batch[idx]
                    stream_list[i].emission  = torch.cat((stream_list[i].emission, emission[idx]), dim=0)
                    stream_list[i].length   += length[idx]
                    transcript, last_blank = greedy_search(stream_list[i].emission)
                    logger.debug(f"[{stream_list[i].id}] text of chunk: {transcript}")
                    stream_list[i].update_stream(transcript, last_blank)

                logger.debug(f"************ [thread-{thread.name}] Decode greedy_search time: {time.time() - start}, batchsize: {len(speeches_batch)}")
            except Exception as inst:
                logger.error(f"Error when decode LM: {inst}")


        return speeches_batch

