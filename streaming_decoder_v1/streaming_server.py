#!/usr/bin/env python3
import os, time, json
import dataclasses
import argparse

import http, ssl
import socket, websockets
import asyncio
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import OmegaConf
from hydra.utils import instantiate
num_thread = int(os.environ["TORCH_THREAD"])

# torch
import torch, torchaudio
torch.set_num_threads(num_thread)
torch.set_num_interop_threads(num_thread)
from speechbrain.inference.speaker import EncoderClassifier

# Resample
import numpy as np
from pydub import AudioSegment as auseg

# local
from http_server import HttpServer
from stream import Stream
from compute_noise import compute_stats_audio
from utils import (
    DecodedResult,
    AudioConfig,
    logger,
    get_hypotheses,
    create_hypotheses,
    load_ngram_endpointing,
)
from lightspeech.models.recognition import greedy_search
import time
from hydra.utils import instantiate
from vad_silero import get_speech_probs, OnnxWrapper



def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="The server will listen on this port",
    )

    parser.add_argument(
        "--max-message-size",
        type=int,
        default=(1 << 20),
        help="""Max message size in bytes.
        The max size per message cannot exceed this limit.
        """,
    )

    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=32,
        help="Max number of messages in the queue for each connection.",
    )

    parser.add_argument(
        "--max-active-connections",
        type=int,
        default=500,
        help="""Maximum number of active connections. The server will refuse
        to accept new connections once the current number of active connections
        equals to this limit.
        """,
    )

    parser.add_argument(
        "--certificate",
        type=str,
        help="""Path to the X.509 certificate. You need it only if you want to
        use a secure websocket connection, i.e., use wss:// instead of ws://.
        You can use sherpa/bin/web/generate-certificate.py
        to generate the certificate `cert.pem`.
        """,
    )

    parser.add_argument(
        "--doc-root",
        type=str,
        default="./web",
        help="""Path to the web root""",
    )

    return parser.parse_args()


class StreamingServer(object):
    def __init__(
        self,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
        doc_root: str,
        certificate: Optional[str] = None,
    ):
        """
        Args:
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          doc_root:
            Path to the directory where files like index.html for the HTTP
            server locate.
          certificate:
            Optional. If not None, it will use secure websocket.
            You can use ./sherpa/bin/web/generate-certificate.py to generate
            it (the default generated filename is `cert.pem`).
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        logger.info(f"NUM_THREAD to loading model: {num_thread}")

        # config
        self.config          = OmegaConf.load("config/asr-online.yaml")

        self.audio_config    = AudioConfig(self.config.audio)

        # model LM
        self.searcher_config = self.config.Linguistic_Model
        self.list_searcher   = {}
        for lm_model in self.searcher_config.keys():
            self.list_searcher[lm_model] = instantiate(self.searcher_config[lm_model], corpus_dir=self.config.corpus_dir)
        logger.info(f"Loaded LM models: {self.searcher_config.keys()}!")

        # model AM
        self.streaming_model = instantiate(self.config.Acoustic_Model, model_dir=self.config.model_dir, device=device)
        self.state_init      = self.streaming_model.init_state()
        logger.info(f"Loaded AM models!")

        # LM endpoiting
        self.ngram, self.prob = load_ngram_endpointing(self.config.LM_Endpointing)
        logger.info(f"Loaded LM endpointing!")

        # VAD Silero
        self.VADSilero_model = OnnxWrapper(self.config.Vad.Silero.model_path, self.config.Vad.Silero.force_onnx_cpu)

        # Speaker Embedding
        self.classifier = EncoderClassifier.from_hparams(source=f"{self.config.Speaker_Diar.model_dir}")
        speaker_audio, _ = torchaudio.load(f"{self.config.Speaker_Diar.model_dir}/{self.config.Speaker_Diar.speaker_wav}")
        self.speaker_emb = self.classifier.encode_batch(speaker_audio)
        self.spk_threshold = self.config.Speaker_Diar.threshold
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        # http server
        self.certificate = certificate
        self.http_server = HttpServer(doc_root)

        self.stream_queue           = asyncio.Queue()
        self.max_message_size       = max_message_size
        self.max_queue_size         = max_queue_size
        self.max_active_connections = max_active_connections

        self.current_active_connections = 0

        self.send_internal = self.config.send_internal
        self.save_audio    = self.config.save_audio
        self.filter_noise  = self.config.filter_noise
        self.noise_threashold = self.config.noise_threashold
        if self.save_audio:
            self.audio_cache = "audio_cache"
            if not os.path.isdir(self.audio_cache): os.makedirs(self.audio_cache)


    async def process_request(
        self,
        path: str,
        request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if "sec-websocket-key" not in request_headers:
            # This is a normal HTTP request
            if path == "/":
                path = "/index.html"
            found, response, mime_type = self.http_server.process_request(path)
            if isinstance(response, str):
                response = response.encode("utf-8")

            if not found:
                status = http.HTTPStatus.NOT_FOUND
            else:
                status = http.HTTPStatus.OK
            header = {"Content-Type": mime_type}
            return status, header, response

        if self.current_active_connections < self.max_active_connections:
            self.current_active_connections += 1
            return None

        # Refuse new connections
        status   = http.HTTPStatus.SERVICE_UNAVAILABLE  # 503
        header   = {"Hint": "The server is overloaded. Please retry later."}
        response = b"The server is busy. Please retry later."

        return status, header, response


    async def run(self, port: int):

        if self.certificate:
            logger.info(f"Using certificate: {self.certificate}")
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.certificate)
        else:
            ssl_context = None
            logger.info("No certificate provided")

        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=500,
            close_timeout=500,
        ):
            ip_list = ["0.0.0.0", "localhost", "127.0.0.1"]
            ip_list.append(socket.gethostbyname(socket.gethostname()))
            proto = "http://" if ssl_context is None else "https://"
            s = "Please visit one of the following addresses:\n\n"
            for p in ip_list:
                s += "  " + proto + p + f":{port}" "\n"
            logger.info(s)

            await asyncio.Future()  # run forever


    async def handle_connection(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        deocoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        try:
            await self.handle_connection_impl(socket)
        except websockets.exceptions.ConnectionClosedError:
            logger.info(f"{socket.remote_address} disconnected")
        finally:
            # Decrement so that it can accept new connections
            self.current_active_connections -= 1

            logger.info(
                f"Disconnected: {socket.remote_address}. "
                f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"  # noqa
            )


    async def handle_connection_impl(
        self,
        socket: websockets.WebSocketServerProtocol,
    ):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        logger.info(
            f"Connected: {socket.remote_address}. "
            f"Number of connections: {self.current_active_connections}/{self.max_active_connections}"
        )
        # Stage 1.1: Init stream pipeline
        stream = Stream(self.config)
        stream.state  = self.state_init
        stream.offset = - (self.audio_config.context_size // self.audio_config.framerate + 1)

        text_all = []
        decoded_result = DecodedResult()
        while True:
            # traceback
            if stream.num_audio_accept == 0:
                noise_previous = torch.Tensor([])

            # Stage 1.2: Get message
            try:
                logger.debug(f"[{stream.id}] Receive message from client!")
                samples = await self.recv_audio_samples(socket)
            except Exception as inst:
                logger.error(f"[{stream.id}] Error when receive message: {inst}")
                break

            if type(samples) is tuple:
                command, argument = samples

                # header
                if command == "__SET_AUDIO_FORMAT__":
                    assert stream.is_eos == False, logger.error("Error: Get header when is_eos is True")
                    stream.id        = argument[0]
                    stream.input_sr  = argument[1]
                    logger.info("Receive header!")
                    if self.save_audio:
                        stream.init_save_audio(f"{self.audio_cache}/{stream.id}.wav")
                    continue

                # end message
                elif command == "__EOS__":
                    logger.info(f"[{stream.id}] Receive EOS message!")
                    stream.is_eos = True

                else:
                    logger.error(f"[{stream.id}] Get unsportted command: {command}")
                    break

            else:
                contain_header = True if b"RIFF" in samples else False
                if stream.input_sr != self.config.audio.sample_rate:
                    start = time.time()
                    samples = auseg(data=samples,
                                sample_width=2,
                                frame_rate=stream.input_sr,
                                channels=1
                            ).set_frame_rate(self.config.audio.sample_rate)
                    samples = samples.get_array_of_samples()
                    fp_arr = np.array(samples).T.astype(np.float32)
                    fp_arr /= np.iinfo(samples.typecode).max
                    samples = torch.from_numpy(fp_arr)

                    logger.debug(f"[{stream.id}] Time resample:{time.time() - start}")
                else:
                    samples = torch.frombuffer(samples, dtype=torch.int16)
                    samples = samples / 32768.0
                if contain_header:
                    samples[:22] = 0

                stream.accept_waveform(waveform=samples)
                logger.debug(f"[{stream.id}] Receive message: {samples.size()}")
                logger.debug(f"[{stream.id}] Total waveform: {stream.audio_total.size(0)/stream.sample_rate}")

            while stream.length_of_segment >= self.audio_config.chunk_length:

                # Stage 1.3: Vad Webrtc
                if not stream.is_contain_token:
                    contain_speech = stream.detect_speech()

                # Stage 1.4: Vad Silero + Decode (AM+LM)
                if contain_speech:
                    # Stage 1: Batching request
                    try:
                        speeches_batch, states_batch = [], []
                        speeches_silero_batch = []
                        index_silero, index_asr = [], []
                        chunk = stream.audio_stream[None, :self.audio_config.chunk_length]
                        if not stream.is_contain_token:
                            index_silero.append(0)
                            stream.n_silero += 1
                            speeches_silero_batch.append(chunk)
                        else:
                            index_asr.append(0)
                            speeches_batch.append(chunk)
                            states_batch.append(stream.state)

                    except Exception as inst:
                        logger.error(f"Error when batching request: {inst}")

                    # State 2: Vad Silero
                    if len(speeches_silero_batch) != 0:
                        try:
                            silero_batch = torch.concat(speeches_silero_batch, dim=0)#[:, self.buffer_length:]
                            start_time = time.time()
                            speeches, speech_probs = get_speech_probs(silero_batch, self.VADSilero_model, sampling_rate=self.audio_config.sample_rate)
                            index_speech, index_nonspeech = np.where(speeches)[0], np.where(~speeches)[0]
                            logger.debug(f"vad silero: {silero_batch.size(0)} {speech_probs} {time.time() - start_time}")
                            logger.debug(f"index_speech: {index_speech}, index_nonspeech: {index_nonspeech}")
                            if len(index_speech) != 0:
                                index_asr.append(0)
                                speeches_batch.append(speeches_silero_batch[0])
                                states_batch.append(stream.state)
                            if len(index_nonspeech) != 0:
                                stream.trailing_blank_duration += 0.64
                                stream.chunk_processed         += 1
                                stream.chunk_processed_total   += 1
                                if stream.emission.size(0) != 0:
                                    stream.offset              += int(stream.segment_size / stream.bias)
                        except Exception as inst:
                            logger.error(f"Error when vad silero: {inst}")

                    # Stage 2: Decode AM
                    start = time.time()
                    if len(speeches_batch) != 0:
                        try:
                            emission, length, states_batch = self.streaming_model.stream(
                                    speeches_batch, self.audio_config.sample_rate, states_batch
                            )
                        except Exception as inst:
                            logger.error(f"Error when decode AM: {inst}")

                        # Stage 3: Decode LM
                        try:
                            start = time.time()
                            for idx, i in enumerate(index_asr):
                                stream.n_decode += 1
                                stream.state = states_batch[idx]
                                stream.emission  = torch.cat((stream.emission, emission[idx]), dim=0)
                                stream.length   += length[idx]
                                transcript, last_blank = greedy_search(stream.emission)
                                logger.debug(f"[{stream.id}] text of chunk: {transcript}")
                                stream.update_stream(transcript, last_blank)

                        except Exception as inst:
                            logger.error(f"Error when decode LM: {inst}")

                        logger.debug(f"Time decode AM_Silero: {time.time()- start}")

                logger.debug(f"[{stream.id}] contain_speech: {contain_speech}, {stream.chunk_processed}, {stream.audio_stream.size()}, {stream.length_of_segment}, {stream.chunk_length}, {stream.offset}, {stream.emission.size()}")

                # Stage 1.5: Check endpointing
                is_final, utt_length = stream.endpoint_detected(self.ngram, self.prob)


                # Stage 1.6: Send internal message
                if self.send_internal and not is_final:
                    text_decode = ""
                    if stream.emission.size(0) != 0:
                        text_decode, _ = greedy_search(stream.emission)

                    if text_decode.strip() != "":
                        hypotheses  = create_hypotheses(text_decode)
                        decoded_result = DecodedResult()
                        decoded_result.result = {
                            "hypotheses": [hypotheses],
                            "final"     : is_final,
                        }
                        try:
                            await socket.send(json.dumps(dataclasses.asdict(decoded_result), ensure_ascii=False))
                            logger.debug(f"[{stream.id}] Decoded result: {json.dumps(dataclasses.asdict(decoded_result), ensure_ascii=False)}")
                        except Exception as inst:
                            logger.error(f"[{stream.id}] Error when send response: {inst}")

                # Stage 1.7: Send final message
                if is_final:
                    start = time.time()
                    # Stage 1.7.1: Rescore hypotheses
                    self.list_searcher[stream.sw_model].decoder.decoder.decode_begin()
                    hypos = self.list_searcher[stream.sw_model].transcript_offline(stream.emission, stream.length, stream.offset)
                    self.list_searcher[stream.sw_model].decoder.decoder.decode_end()
                    stream.emission   = torch.Tensor([])
                    stream.length     = 0

                    hypotheses  = get_hypotheses(hypos)

                    decoded_result = DecodedResult()
                    decoded_result.id             = stream.id
                    decoded_result.segment_length = utt_length

                    decoded_result = self._update_decoded_result(stream, decoded_result, hypotheses)

                    text_decode = hypotheses["transcript"]
                    text_all.append(text_decode)
                    logger.debug(f"[{stream.id}] Decode LM time : {time.time() - start}")

                    # Stage 1.7.2: Reinit state of searcher
                    stream.state = self.state_init

                    # Stage 1.7.3: Compute noise
                    if text_decode.strip() != "":
                        try:
                            decoded_result, noise_previous = compute_stats_audio(stream.audio_total, stream.offset_compute_stats, noise_previous, decoded_result, sr=stream.sample_rate)
                            _is_same_spk = self._verify_speaker(stream.audio_total, stream.offset_compute_stats, decoded_result, stream.sample_rate)
                            decoded_result.is_speaker = _is_same_spk
                            if decoded_result.vol_speech <= self.noise_threashold:
                                if self.filter_noise:
                                    logger.debug(f"Filter out segment with small volume: {decoded_result.vol_speech}")
                                    continue
                            await socket.send(json.dumps(dataclasses.asdict(decoded_result), ensure_ascii=False))
                            logger.info(f"[{stream.id}] Decoded result: {json.dumps(dataclasses.asdict(decoded_result), ensure_ascii=False)}")
                        except Exception as inst:
                            logger.error(f"[{stream.id}] Error when send response: {inst}")
                    stream.discard_decoded_segment(utt_length, self.save_audio)

            # Stage 1.8: Send final message when EOS
            if stream.is_eos:
                assert stream.audio_stream.size(0) <= stream.chunk_length, \
                    logger.error(f"[{stream.id}] Error: Size of final segment less than chunk length: \
                        {stream.audio_stream.size(0)} and {stream.chunk_length}!")

                # Stage 1.8.3 Rescore hypotheses
                if stream.emission.size(0) != 0:
                    self.list_searcher[stream.sw_model].decoder.decoder.decode_begin()
                    hypos = self.list_searcher[stream.sw_model].transcript_offline(stream.emission, stream.length, stream.offset)
                    self.list_searcher[stream.sw_model].decoder.decoder.decode_end()

                    hypotheses  = get_hypotheses(hypos)

                    decoded_result.segment_length = ( stream.chunk_processed       * stream.segment_length  ) / stream.sample_rate

                    decoded_result = self._update_decoded_result(stream, decoded_result, hypotheses)

                    text_decode = hypotheses["transcript"]
                    text_all.append(text_decode)

                    # Stage 1.8.4 Compute noise and send final message
                    if text_decode.strip() != "":
                        logger.info(f"[{stream.id}] Return message final: {decoded_result}")
                        decoded_result, noise_previous = compute_stats_audio(stream.audio_total, stream.offset_compute_stats, noise_previous, decoded_result, sr=stream.sample_rate)
                        await socket.send(json.dumps(dataclasses.asdict(decoded_result), ensure_ascii=False))

                text_all = " ".join(text_all)
                logger.info(f"[{stream.id}] Text decode all: {text_all}")
                logger.info(f"[{stream.id}] Time decode all request: {time.time() - stream.start_time}")
                logger.info(f"[{stream.id}] Stats: {stream.chunk_processed_total}, {stream.n_webrtc}, {stream.n_silero}, {stream.n_decode}")
                logger.info(f"[{stream.id}] Return message: __REQUEST_COMPLETED__")
                await socket.send("__REQUEST_COMPLETED__")

                # Reinit Stream
                text_all = []
                stream = Stream(self.config)
                stream.state  = self.state_init
                stream.offset = - (self.audio_config.context_size // self.audio_config.framerate + 1)

    def _verify_speaker(self, audio_total, offset, decoded_result: DecodedResult, sr: int):
        start = time.time()
        word_start  = int((decoded_result.word_start - offset) * sr)
        word_end    = int((decoded_result.word_end - offset) * sr)
        speech      = audio_total[word_start: word_end]
        segment_emb = self.classifier.encode_batch(speech)
        score = self.similarity(self.speaker_emb, segment_emb)
        logger.info(f"Speaker Score: {score} - {time.time() - start}")
        if score > self.spk_threshold:
            return True
        else: 
            return False

    def _update_decoded_result(self, stream: Stream, decoded_result: DecodedResult, hypotheses: dict):
        decoded_result.segment        = stream.segment
        decoded_result.result         = {
            "hypotheses": [hypotheses],
            "final"     : True,
        }
        decoded_result.total_length   = ( stream.chunk_processed_total * stream.segment_length ) / stream.sample_rate
        if len(hypotheses["word_alignment"]) != 0:
            decoded_result.segment_start = round(decoded_result.total_length - decoded_result.segment_length, 2)
            decoded_result.word_start    = hypotheses["word_alignment"][0]["start"]
            decoded_result.word_end      = round(hypotheses["word_alignment"][-1]["start"] + hypotheses["word_alignment"][-1]["length"], 2)
        return decoded_result


    async def recv_audio_samples(
        self,
        socket: websockets.WebSocketServerProtocol,
    ) -> Optional[torch.Tensor]:
        """Receives byte message from the client.

        Each message contains either a bytes buffer containing audio samples
        in 8 kHz or contains "EOS" meaning the end of utterance.

        Args:
          socket:
            The socket for communicating with the client.
        Returns:
          Return a byte message or return (command, argument).
        """
        message = await socket.recv()

        if type(message) == str:
            message = json.loads(message)
            logger.info(f"Receiclskdlkas: {message}")
            if message["__COMMAND__"] == "__EOS__":
                return "__EOS__", None
            elif message["__COMMAND__"] == "__SET_AUDIO_FORMAT__":
                sr = json.loads(message["__ARGUMENT__"])["sample_rate"]
                logger.info(f"Header of request: {message, sr}")
                return "__SET_AUDIO_FORMAT__", (message["request-id"], sr)
        return message


@torch.no_grad()
def main():
    args = get_args()

    port = args.port
    max_message_size = args.max_message_size
    max_queue_size = args.max_queue_size
    max_active_connections = args.max_active_connections
    certificate = args.certificate
    doc_root = args.doc_root

    if certificate and not Path(certificate).is_file():
        raise ValueError(f"{certificate} does not exist")

    if not Path(doc_root).is_dir():
        raise ValueError(f"Directory {doc_root} does not exist")


    server = StreamingServer(
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
        certificate=certificate,
        doc_root=doc_root,
    )
    # asyncio.run(server.run(port))
    asyncio.get_event_loop().run_until_complete(server.run(port))
    asyncio.get_event_loop().run_forever()




# See https://github.com/pytorch/pytorch/issues/38342
# and https://github.com/pytorch/pytorch/issues/33354
#
# If we don't do this, the delay increases whenever there is
# a new request that changes the actual batch size.
# If you use `py-spy dump --pid <server-pid> --native`, you will
# see a lot of time is spent in re-compiling the torch script model.
# torch._C._jit_set_profiling_executor(False)
# torch._C._jit_set_profiling_mode(False)
# torch._C._set_graph_executor_optimize(False)
"""
// Use the following in C++
torch::jit::getExecutorMode() = false;
torch::jit::getProfilingMode() = false;
torch::jit::setGraphExecutorOptimize(false);
"""

if __name__ == "__main__":
    main()
