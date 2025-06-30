import logging, re, os
import requests
from dataclasses import dataclass, field
from typing import Dict
from omegaconf import DictConfig
from logging import handlers, Logger


class AudioConfig(object):
    def __init__(self, config: DictConfig):
        """
        config: Audio config
        """
        # audio config
        self.sample_rate    = config.sample_rate
        self.hop_length     = int(config.hop_length * config.sample_rate)
        self.segment_size   = config.segment_size
        self.segment_length = self.segment_size * self.hop_length
        self.context_size   = config.context_size
        self.bias           = config.bias
        self.buffer_length  = int((self.context_size + self.bias) * self.hop_length)
        self.chunk_length   = self.segment_length + self.buffer_length
        self.framerate      = config.framerate


@dataclass
class DecodedResult:
    id:             str = field(default_factory=str)
    status:         int = field(default_factory=int)
    msg:            int = field(default_factory=int)
    segment:        int = field(default_factory=int)
    result:         Dict[str,float] = field(default_factory=str)
    segment_start:  float = field(default_factory=float)
    segment_length: float = field(default_factory=float)
    total_length:   float = field(default_factory=float)
    message_type:   int   = field(default_factory=int)
    word_start:     float = field(default_factory=float)
    word_end:       float = field(default_factory=float)
    snr:            float = 0.0
    vol_noise:      float = 0.0
    vol_speech:     float = 0.0
    is_speaker:     bool = False


@dataclass
class AudacitySegment:
    start_time: float = field(default_factory=float)
    stop_time:  float = field(default_factory=float)
    label:      str   = field(default_factory=str)


url = f'http://localhost:{os.environ["NORM_PORT"]}/cleanoutput'


def request(text):
    r = requests.post(url=url, data={'text': text})
    return r.text.replace('phantram', '%')


def convert2audacity(data: list[AudacitySegment], output_file: str):
    f = open(output_file, "w")
    for segment in data:
        start_time = str(segment.start_time)
        stop_time  = str(segment.stop_time)
        label      = segment.label
        f.write("\t".join([start_time, stop_time, label]) + "\n")
    f.close()


def setup_logger(
    use_console: bool = True,
) -> Logger:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    formatter = (
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    debug_handler = handlers.RotatingFileHandler(
        'logs/debug.log',
        maxBytes=500*1024**2, backupCount=5
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_formater = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s")
    debug_handler.setFormatter(debug_formater)

    logger.addHandler(debug_handler)

    if use_console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(formatter))
        logger.addHandler(console)
    return logger
logger = setup_logger()


def load_ngram_endpointing(lm=""):
    ngram=4
    prob={}
    with open(lm,"r") as f:
        lm_all=f.read().split("\n")
        for line in lm_all:
            try:
                line=line.split("\t")
                prob[line[1]]=float(line[0])
            except:
                try:
                    temp=re.findall("ngram [0-9]*=", line[0])[0]
                    ngram=int(re.findall("[0-9]+",temp)[0])
                except:
                    continue
    return ngram, prob

def compute_relative_cost(utt, ngram, prob):
    end_utt=("<s> "+utt).split()[1-ngram:]

    end_utt.append("</s>")
    final_cost=float('inf')
    while True:
        try:
            final_cost=prob[" ".join(end_utt)]
            break
        except:
            end_utt.pop(0)

    relative_cost = -5*final_cost
    return relative_cost


def create_hypotheses(transcript):
    hypotheses = {}

    hypotheses["transcript"]            = transcript
    hypotheses["transcript_normalized"] = transcript
    hypotheses["confidence"]            = 0.0
    hypotheses["likelihood"]            = 1.0
    hypotheses["word_alignment"]        = []

    return hypotheses


def get_hypotheses(decoded_result):
    hypotheses = {}
    word_alignments     = []
    word_confident_list = []
    word_hyp_list       = []
    for part in decoded_result:
        r = {}
        r["word"]       = part["word"].replace("<<", "").replace(">>", "")
        r["start"]      = part["beg"]
        r["length"]     = round(part["end"] - part["beg"], 2)
        r["confidence"] = part["confidence"]
        word_confident_list.append(part["confidence"])
        word_hyp_list.append(r["word"])
        word_alignments.append(r)

    if len(word_confident_list) == 0:
        sent_confidence = 0
    else:
        sent_confidence = sum(word_confident_list)/len(word_confident_list)

    transcript                          = " ".join(word_hyp_list)

    hypotheses["transcript"]            = transcript
    hypotheses["transcript_normalized"] = request(transcript)
    hypotheses["confidence"]            = round(sent_confidence, 2)
    hypotheses["word_alignment"]        = word_alignments

    return hypotheses


def get_hypotheses_en(transcript):
    hypotheses = {}
    hypotheses["transcript"]            = transcript
    hypotheses["transcript_normalized"] = transcript
    return hypotheses
