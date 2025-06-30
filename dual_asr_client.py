#!/usr/bin/env python3
from urllib.parse import urlparse
from websocket import WebSocketApp, ABNF
import numpy as np
import threading
import multiprocessing as mp
import socket
import wave
import json
import time
import sys
import argparse
import subprocess
from scipy.signal import resample
from dataclasses import dataclass, asdict

# Server URLs
DEFAULT_VI_ASR_URL = "ws://localhost:9432"
DEFAULT_EN_ASR_URL = "ws://localhost:9433"

# Audio settings
BUFFER_SIZE = 8000
SAMPLE_RATE = 16000  # 16000 Hz

# ANSI colors for terminal output
COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'RESET': '\033[0m'
}

def clear_line():
    """Clear the current line in the terminal"""
    print('\r\033[K', end="", flush=True)

def write(text, color=COLORS['RESET']):
    """Write text to terminal with specified color"""
    clear_line()
    print(f"{color}{text}{COLORS['RESET']}", end="", flush=True)

def list_audio_devices():
    """List all available audio devices"""
    try:
        result = subprocess.run(["arecord", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Available audio devices:")
        print(result.stdout.decode())
    except Exception as e:
        print(f"Error listing audio devices: {e}")

class DualAsrClient:
    """Client that connects to both Vietnamese and English ASR servers"""
    
    def __init__(self,
                 buffer_size=BUFFER_SIZE,
                 sample_rate=SAMPLE_RATE,
                 vi_asr_url=DEFAULT_VI_ASR_URL,
                 en_asr_url=DEFAULT_EN_ASR_URL,
                 output_wav_file=None,
                 audio_device=None,
                 use_pulse=False,
                 input_file=None):
        
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.input_file = input_file
        
        # Prepare WebSocket URLs with parameters
        self.vi_asr_url = f"{vi_asr_url}/voice/api/asr/v1/ws/decode_online?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int){self.sample_rate}"
        self.en_asr_url = f"{en_asr_url}/voice/api/asr/v1/ws/decode_online?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int){self.sample_rate}"
        
        self.audio_device = audio_device
        self.use_pulse = use_pulse
        self.output_wav_file = output_wav_file
        self.wav_file = None
        
        # Audio processing objects
        self.audio_process = None
        self.shutdown_event = mp.Event()
        self.audio_queue_vi = mp.Queue()
        self.audio_queue_en = mp.Queue()
        self.is_running = True
        
        # Text results
        self.vi_realtime_text = ""
        self.vi_final_text = ""
        self.en_realtime_text = ""
        self.en_final_text = ""
        
        # WebSocket connections
        self.vi_ws = None
        self.en_ws = None
        self.vi_ws_thread = None
        self.en_ws_thread = None
        
        # Connect to both servers
        vi_connected = self.connect_vi()
        # en_connected = False  # Bỏ qua máy chủ tiếng Anh
        # if en_asr_url and en_asr_url != "":
        en_connected = self.connect_en()
        
        if not (vi_connected or en_connected):
            print("Failed to connect to any ASR server.", file=sys.stderr)
            return
        
        # Initialize audio streaming
        if self.input_file:
            self.start_playback()
        else:
            if not self.initialize_audio_stream():
                print("Failed to initialize audio recording.", file=sys.stderr)
                return
            self.start_recording()
    
    def initialize_audio_stream(self):
        """Initialize audio recording using arecord"""
        cmd = ["arecord"]
        
        # Add device selection if specified
        if self.use_pulse:
            cmd.extend(["-D", "pulse"])
        elif self.audio_device:
            cmd.extend(["-D", self.audio_device])
        
        # Add format parameters
        cmd.extend(["-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1", "-t", "raw"])
        # Command explanation:
        # arecord: Linux utility for sound recording
        # -D <device>: Specifies audio device to use
        #   - "pulse" uses PulseAudio sound server
        #   - Or specific hardware device like "hw:0,0"
        # -d 5: Record duration of 5 seconds
        # -f S16_LE: 16-bit signed little-endian format
        # -r 16000: Sample rate of 16kHz
        # -c 1: Single channel (mono) recording
        # -t raw: Output raw audio data without headers
        # Show verbose output for debug
        cmd.append("-v")
        
        try:
            print(f"Starting audio recording with command: {' '.join(cmd)}")
            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Check if the process is running
            if self.audio_process.poll() is not None:
                stderr_output = self.audio_process.stderr.read().decode()
                print(f"Audio recording failed to start: {stderr_output}")
                return False
                
            return True
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            return False
    
    def is_server_running(self, url):
        """Check if the ASR server is running at the specified URL"""
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        port = parsed_url.port or 80
        try:
            print(f"Checking if server is running at {host}:{port}...")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)  # Set timeout to 2 seconds
                result = s.connect_ex((host, port))
                if result == 0:
                    print(f"Server at {host}:{port} is running")
                    return True
                else:
                    print(f"Server at {host}:{port} is not running (error code: {result})")
                    return False
        except Exception as e:
            print(f"Error checking server at {url}: {e}")
            return False
    
    def connect_vi(self):
        """Connect to Vietnamese ASR server"""
        if not self.is_server_running(DEFAULT_VI_ASR_URL):
            print(f"Vietnamese ASR server is not running at {DEFAULT_VI_ASR_URL}", file=sys.stderr)
            return False
        
        try:
            self.vi_ws = WebSocketApp(
                self.vi_asr_url,
                on_message=self.on_vi_message,
                on_error=self.on_vi_error,
                on_close=self.on_vi_close,
                on_open=self.on_vi_open
            )
            
            self.vi_ws_thread = threading.Thread(target=self.vi_ws.run_forever)
            self.vi_ws_thread.daemon = False
            self.vi_ws_thread.start()
            
            # Give the connection a moment to establish
            time.sleep(0.5)
            
            print(f"Connected to Vietnamese ASR server at {DEFAULT_VI_ASR_URL}")
            return True
        except Exception as e:
            print(f"Error connecting to Vietnamese ASR server: {e}")
            return False
    
    def connect_en(self):
        """Connect to English ASR server"""
        print(f"Attempting to connect to English ASR server at {DEFAULT_EN_ASR_URL}")
        
        if not self.is_server_running(DEFAULT_EN_ASR_URL):
            print(f"English ASR server is not running at {DEFAULT_EN_ASR_URL}", file=sys.stderr)
            return False
        
        try:
            print(f"English ASR server is running, establishing WebSocket connection to {self.en_asr_url}")
            self.en_ws = WebSocketApp(
                self.en_asr_url,
                on_message=self.on_en_message,
                on_error=self.on_en_error,
                on_close=self.on_en_close,
                on_open=self.on_en_open
            )
            
            self.en_ws_thread = threading.Thread(target=self.en_ws.run_forever)
            self.en_ws_thread.daemon = False
            self.en_ws_thread.start()
            
            # Give the connection a moment to establish
            time.sleep(0.5)
            
            print(f"Connected to English ASR server at {DEFAULT_EN_ASR_URL}")
            return True
        except Exception as e:
            print(f"Error connecting to English ASR server: {e}")
            return False
    
    def start_recording(self):
        """Start recording audio and sending to ASR servers"""
        self.recording_thread = threading.Thread(target=self.record_and_send_audio)
        self.recording_thread.daemon = False
        self.recording_thread.start()
    
    def record_and_send_audio(self):
        """Record audio data and send to both ASR servers"""
        try:
            if self.output_wav_file and not self.wav_file:
                self.wav_file = wave.open(self.output_wav_file, 'wb')
                self.wav_file.setnchannels(1)
                self.wav_file.setsampwidth(2)
                self.wav_file.setframerate(self.sample_rate)
            
            print("Recording and sending audio to ASR servers...")
            print("Speak into the microphone now...")
            
            chunk_count = 0
            while self.is_running:
                chunk = self.audio_process.stdout.read(self.buffer_size)
                if not chunk:
                    print("No audio data received, check your microphone")
                    time.sleep(1)
                    continue
                
                if self.wav_file:
                    self.wav_file.writeframes(chunk)
                
                # Send the audio chunk to both queues
                if self.vi_ws:
                    self.audio_queue_vi.put_nowait(chunk)
                if self.en_ws:
                    self.audio_queue_en.put_nowait(chunk)
                
                # Periodically print status (every 100 chunks ~ 5 seconds at 16kHz)
                chunk_count += 1
                if chunk_count % 100 == 0:
                    vi_queue_size = self.audio_queue_vi.qsize() if self.vi_ws else "N/A"
                    en_queue_size = self.audio_queue_en.qsize() if self.en_ws else "N/A"
                    print(f"Audio status - chunks sent: {chunk_count}, "
                          f"VI queue: {vi_queue_size}, EN queue: {en_queue_size}")
            
        except Exception as e:
            print(f"Error recording audio: {e}")
        finally:
            self.cleanup_audio()
            self.is_running = False
    
    def cleanup_audio(self):
        """Clean up audio resources"""
        if self.audio_process:
            self.audio_process.terminate()
            self.audio_process = None
        
        if self.wav_file:
            self.wav_file.close()
            self.wav_file = None
    
    # Vietnamese ASR callback handlers
    def on_vi_message(self, ws, message):
        try:
            data = json.loads(message)
            # For debugging, print raw message type occasionally
            if 'msg' in data and data['msg'] not in [0, 1, 2]:  # Skip common message types to reduce noise
                print(f"[VI-DEBUG] Received message type: {data['msg']}")
            
            if not 'result' in data:
                # Uncomment the line below if you want to see non-result messages
                # print(f"[VI-DEBUG] Received non-result message: {message[:100]}...")
                return
            
            if data['result']['final']:
                transcript = data["result"]["hypotheses"][0]["transcript_normalized"]
                self.vi_final_text += transcript[0].upper() + transcript[1:] + ". "
                
                # Check if from the target speaker
                is_speaker = data.get("is_speaker", True)
                
                # Display text with appropriate color
                print("\n[VI-FINAL] ", end="")
                if is_speaker:
                    write(self.vi_final_text, COLORS['GREEN'])
                else:
                    write(self.vi_final_text, COLORS['RED'])
            else:
                transcript = data["result"]["hypotheses"][0]["transcript"]
                if transcript != self.vi_realtime_text and transcript.strip():
                    self.vi_realtime_text = transcript
                    print("\n[VI-REALTIME] ", end="")
                    write(self.vi_realtime_text, COLORS['YELLOW'])
        except Exception as e:
            print(f"Error processing Vietnamese ASR message: {message[:50]}... Error: {e}")
    
    def on_vi_error(self, ws, error):
        print(f"Vietnamese ASR WebSocket error: {error}")
        if isinstance(error, Exception):
            print(f"  Error type: {type(error).__name__}")
            print(f"  Error details: {str(error)}")
    
    def on_vi_close(self, ws, close_status_code, close_msg):
        print(f"Vietnamese ASR connection closed (Status code: {close_status_code}, Message: {close_msg})")
    
    def on_vi_open(self, ws):
        def run():
            """Thread function to send audio data to Vietnamese ASR server"""
            while self.is_running:
                try:
                    if not self.audio_queue_vi.empty():
                        chunk = self.audio_queue_vi.get(block=False)
                        if self.vi_ws:
                            try:
                                self.vi_ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
                            except Exception as e:
                                print(f"Error sending to Vietnamese ASR: {e}")
                    else:
                        time.sleep(0.01)
                except Exception as e:
                    print(f"Error in Vietnamese sender thread: {e}")
                    time.sleep(0.1)
        
        audio_sender_thread = threading.Thread(target=run)
        audio_sender_thread.daemon = True
        audio_sender_thread.start()
    
    # English ASR callback handlers
    def on_en_message(self, ws, message):
        try:
            data = json.loads(message)
            # For debugging, print raw message type 
            if 'msg' in data:
                print(f"[EN-DEBUG] Received message type: {data['msg']}")
            
            if not 'result' in data:
                print(f"[EN-DEBUG] Received non-result message: {message[:100]}...")
                return
            
            if data['result']['final']:
                transcript = data["result"]["hypotheses"][0]["transcript"]
                self.en_final_text += transcript[0].upper() + transcript[1:] + ". "
                
                # Check if from the target speaker
                is_speaker = data.get("is_speaker", True)
                
                # Display text with appropriate color
                print("\n[EN-FINAL] ", end="")
                if is_speaker:
                    write(self.en_final_text, COLORS['GREEN'])
                else:
                    write(self.en_final_text, COLORS['GREEN'])
            else:
                transcript = data["result"]["hypotheses"][0]["transcript"]
                if transcript != self.en_realtime_text and transcript.strip():
                    self.en_realtime_text = transcript
                    print("\n[EN-REALTIME] ", end="")
                    write(self.en_realtime_text, COLORS['BLUE'])
        except Exception as e:
            print(f"Error processing English ASR message: {message[:50]}... Error: {e}")
    
    def on_en_error(self, ws, error):
        print(f"English ASR WebSocket error: {error}")
        if isinstance(error, Exception):
            print(f"  Error type: {type(error).__name__}")
            print(f"  Error details: {str(error)}")
    
    def on_en_close(self, ws, close_status_code, close_msg):
        print(f"English ASR connection closed (Status code: {close_status_code}, Message: {close_msg})")
    
    def on_en_open(self, ws):
        def run():
            """Thread function to send audio data to English ASR server"""
            while self.is_running:
                try:
                    if not self.audio_queue_en.empty():
                        chunk = self.audio_queue_en.get(block=False)
                        if self.en_ws:
                            try:
                                self.en_ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
                            except Exception as e:
                                print(f"Error sending to English ASR: {e}")
                    else:
                        time.sleep(0.01)
                except Exception as e:
                    print(f"Error in English sender thread: {e}")
                    time.sleep(0.1)
        
        audio_sender_thread = threading.Thread(target=run)
        audio_sender_thread.daemon = True
        audio_sender_thread.start()
    
    def shutdown(self):
        """Shutdown all resources"""
        print("\nShutting down ASR client...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Close WebSocket connections
        if self.vi_ws:
            self.vi_ws.close()
        
        if self.en_ws:
            self.en_ws.close()
        
        # Join threads
        if self.vi_ws_thread:
            self.vi_ws_thread.join(timeout=1)
        
        if self.en_ws_thread:
            self.en_ws_thread.join(timeout=1)
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1)
        
        # Clean up audio resources
        self.cleanup_audio()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    def start_playback(self):
        """Start playing audio file and sending to ASR servers"""
        self.playback_thread = threading.Thread(target=self.play_and_send_audio)
        self.playback_thread.daemon = False
        self.playback_thread.start()
        
    def play_and_send_audio(self):
        """Play audio file and send data to ASR servers"""
        try:
            print(f"Playing audio file: {self.input_file}")
            
            # Open the input WAV file
            with wave.open(self.input_file, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                
                print(f"File info: {sample_rate}Hz, {sample_width} bytes/sample, {channels} channel(s)")
                
                # Open output file if specified
                if self.output_wav_file:
                    self.wav_file = wave.open(self.output_wav_file, 'wb')
                    self.wav_file.setnchannels(1)
                    self.wav_file.setsampwidth(2)
                    self.wav_file.setframerate(self.sample_rate)
                
                # Read and process the audio data
                while self.is_running:
                    chunk = wav_file.readframes(self.buffer_size // (sample_width * channels))
                    if not chunk:
                        print("End of file reached")
                        break
                    
                    # Convert to mono if stereo
                    if channels == 2:
                        # Convert bytes to numpy array
                        audio_data = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 2)
                        # Take average of two channels
                        mono_data = (audio_data[:, 0] + audio_data[:, 1]) // 2
                        chunk = mono_data.astype(np.int16).tobytes()
                    
                    # Resample if needed
                    if sample_rate != self.sample_rate:
                        audio_array = np.frombuffer(chunk, dtype=np.int16)
                        num_samples = int(len(audio_array) * self.sample_rate / sample_rate)
                        resampled = resample(audio_array, num_samples)
                        chunk = resampled.astype(np.int16).tobytes()
                    
                    if self.wav_file:
                        self.wav_file.writeframes(chunk)
                    
                    # Send the audio chunk to the queues
                    if self.vi_ws:
                        self.audio_queue_vi.put_nowait(chunk)
                    if self.en_ws:
                        self.audio_queue_en.put_nowait(chunk)
                    
                    # Add a small delay to simulate real-time streaming
                    time.sleep(len(chunk) / (2 * self.sample_rate))  # 2 bytes per sample
                
                print("Finished sending audio file")
                
        except Exception as e:
            print(f"Error processing audio file: {e}")
        finally:
            self.cleanup_audio()
            # Keep WebSocket open to receive final results
            time.sleep(2)
            self.is_running = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dual language ASR client for Vietnamese and English")
    
    parser.add_argument("--vi-url", default=DEFAULT_VI_ASR_URL,
                        help=f"Vietnamese ASR server URL (default: {DEFAULT_VI_ASR_URL})")
    
    parser.add_argument("--en-url", default=DEFAULT_EN_ASR_URL,
                        help=f"English ASR server URL (default: {DEFAULT_EN_ASR_URL})")
    
    parser.add_argument("--device", default=None,
                        help="Audio device to use (e.g., hw:1,0). Leave empty to use default device")
    
    parser.add_argument("--pulse", action="store_true",
                        help="Use PulseAudio instead of specific hardware device")
    
    parser.add_argument("--output", default=None,
                        help="Output WAV file to save the recorded audio (optional)")
    
    parser.add_argument("--input-file", default=None,
                        help="Input audio file to use instead of recording from microphone")
    
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        # List audio devices if requested
        if args.list_devices:
            list_audio_devices()
            sys.exit(0)
        
        # Create and start the dual ASR client
        with DualAsrClient(vi_asr_url=args.vi_url,
                         en_asr_url=args.en_url,
                         audio_device=args.device,
                         use_pulse=args.pulse,
                         output_wav_file=args.output,
                         input_file=args.input_file) as client:
            
            print("\nDual ASR client is running. Press Ctrl+C to exit.")
            
            # Keep the main thread alive
            while client.is_running:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 