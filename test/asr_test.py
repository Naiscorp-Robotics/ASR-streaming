#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
import json
import time
import websocket
from pathlib import Path
from threading import Thread
from datetime import datetime

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class AsrWebSocket:
    def __init__(self, sample_rate=16000, speed=1, interval=4):

        self.sample_rate = sample_rate
        self.speed = speed
        self.interval = interval
        self.audio_stream = None
        self.ws = None
        self.is_closed = False
        self.send_count = 0
        self.start_time = 0


    def on_open(self, ws):
        def run():
            logging.info(f"Time connect: {time.time() - self.time_create}")
            self.start_time = time.time()
            count = 0
            chunk_size = self.sample_rate * 2 * self.speed // self.interval
            data_size  = len(self.audio_stream)
            num_chunk  = round(data_size/chunk_size)
            for i in range(num_chunk):
                block = self.audio_stream[i*chunk_size: (i+1)*chunk_size]
                if self.is_closed:
                    count += 1
                else:
                    self.send_data(ws, block)
            if count > 0:
                logging.info("Discarded ?lock" %count)
            if not self.is_closed:
                ws.send('EOS')
                time.sleep(0.3)

        my_thread = Thread(target=run)
        my_thread.start()


    def send_data(self, ws, data):
        if self.is_closed:
            return
        send_time = time.time() - self.start_time
        self.send_count += 1
        if send_time < self.send_count/self.interval:
            time.sleep(self.send_count/self.interval - send_time)
        ws.send(data, opcode=0x2)


    def on_message(self, ws, message):
        utt = json.loads(message)
        if not 'result' in utt:
            return
        elif utt['result']['final']:
            results = json.loads(message)
            logging.info(results["is_speaker"])
            try:
                is_speaker = results["is_speaker"]
            except:
                is_speaker = True
            if is_speaker:
                logging.info('\033[92m' + results["result"]["hypotheses"][0]["transcript_normalized"])
            else:
                logging.info('\033[91m' + results["result"]["hypotheses"][0]["transcript_normalized"])
        elif not utt['result']['final']:
            results = json.loads(message)
            logging.info('\033[93m' + results["result"]["hypotheses"][0]["transcript"])

    def on_error(self, ws, error):
        self.is_closed=True
        logging.info(error)

    def on_close(self, ws, close_status_code, close_msg):
        stop_time = time.time()
        elapsed = stop_time - self.start_time;
        logging.info(f"Total run time: {elapsed}")
        self.is_closed = True
    def recognize(self, audio_stream):
        import ssl
        self.audio_stream = audio_stream
        self.time_create = time.time()
        ws = websocket.WebSocketApp('ws://localhost:9432',
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close,
                                         on_open=self.on_open)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

if __name__ == "__main__":

    import sys
    audio_path = sys.argv[1]
    ws_asr = AsrWebSocket()
    f = open(audio_path, 'rb')
    f.read(44)
    data = f.read()
    ws_asr.recognize(data)
