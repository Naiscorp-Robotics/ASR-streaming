#!/usr/bin/env python3
# Minimal ASR client - stripped down to the absolute essentials

import threading
import time
import json
import sys
import subprocess
import socket
from urllib.parse import urlparse
from websocket import WebSocketApp, ABNF

# Constants - modify these if needed
ASR_URL = "ws://localhost:9432" # For Vietnamese - use 9433 for English
BUFFER_SIZE = 8000
SAMPLE_RATE = 16000

def main():
    """Main function to run the minimal ASR client"""
    
    # First check if the server is running
    parsed_url = urlparse(ASR_URL)
    host = parsed_url.hostname
    port = parsed_url.port or 80
    
    # Test socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) != 0:
            print(f"Server at {host}:{port} is not running!")
            return
    
    print(f"Server at {host}:{port} is running, connecting...")
        
    # Prepare URL with content-type
    content_type_param = f"?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int){SAMPLE_RATE},+format=(string)S16LE,+channels=(int)1"
    ws_url = f"{ASR_URL}/voice/api/asr/v1/ws/decode_online{content_type_param}"
    
    print(f"Connecting to {ws_url}")
    
    # Global flags
    is_running = True
    audio_queue = []
    
    # Message handler
    def on_message(ws, message):
        try:
            data = json.loads(message)
            print(f"\nReceived message: {message}")
            
            # Process and display recognition results
            if 'result' in data and 'hypotheses' in data['result'] and data['result']['hypotheses']:
                is_final = data['result'].get('final', False)
                transcript = data['result']['hypotheses'][0].get('transcript', '')
                if is_final:
                    print(f"\n[FINAL] {transcript}")
                else:
                    print(f"\n[INTERIM] {transcript}")
        except Exception as e:
            print(f"Error processing message: {e}")
            print(f"Raw message: {message}")
    
    # Start audio recording
    def record_audio():
        nonlocal audio_queue
        cmd = ["arecord", "-D", "default", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]
        
        try:
            print("Starting audio recording...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            while is_running:
                # Read audio chunks
                chunk = process.stdout.read(BUFFER_SIZE)
                if not chunk:
                    break
                audio_queue.append(chunk)
                
                # Print a dot to show activity
                if len(audio_queue) % 10 == 0:
                    print(".", end="", flush=True)
                    
        except Exception as e:
            print(f"Error recording audio: {e}")
        finally:
            process.terminate()
            print("Audio recording stopped")
    
    # Send audio data
    def send_audio(ws):
        nonlocal audio_queue
        last_sent = 0
        
        print("Starting audio transmission...")
        while is_running:
            try:
                if len(audio_queue) > last_sent:
                    chunk = audio_queue[last_sent]
                    ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
                    last_sent += 1
                    
                    # Print activity marker
                    if last_sent % 10 == 0:
                        print(">", end="", flush=True)
                else:
                    # No new data, wait a bit
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error sending audio: {e}")
                break
    
    # WebSocket events
    def on_open(ws):
        print("WebSocket connection established!")
        
        # Start audio recording in a separate thread
        threading.Thread(target=record_audio, daemon=True).start()
        
        # Start sending audio in another thread
        threading.Thread(target=send_audio, args=(ws,), daemon=True).start()
    
    def on_error(ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        nonlocal is_running
        is_running = False
    
    # Create WebSocket connection
    ws = WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run WebSocket connection in the main thread
    try:
        print("Starting WebSocket connection...")
        ws.run_forever()
    except KeyboardInterrupt:
        print("\nStopping the application...")
    finally:
        is_running = False

if __name__ == "__main__":
    main() 