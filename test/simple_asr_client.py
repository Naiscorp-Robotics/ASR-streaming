from urllib.parse import urlparse
import numpy as np
import threading
import subprocess
import socket
import json
import time
import sys
import os
from websocket import WebSocketApp, ABNF
from datetime import datetime

# Constants
VI_ASR_URL = "ws://localhost:9432"
EN_ASR_URL = "ws://localhost:9433"
BUFFER_SIZE = 8000
SAMPLE_RATE = 16000
DEBUG = True  # Set to True to see debug messages

def debug_print(message):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        print(f"[DEBUG] {message}")

def get_timestamp():
    """Return current timestamp as HH:MM:SS.mmm"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def clear_line():
    """Clear current console line"""
    print('\r\033[K', end="", flush=True)

class SimpleAsrClient:
    def __init__(self, 
                 buffer_size=BUFFER_SIZE, 
                 sample_rate=SAMPLE_RATE,
                 vi_asr_url=VI_ASR_URL,
                 en_asr_url=EN_ASR_URL,
                 test_mode=False):
                 
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.test_mode = test_mode
        
        # Base URLs
        self.vi_base_url = vi_asr_url
        self.en_base_url = en_asr_url
        
        # Content type parameters - use exact format from original client
        self.content_type_param = f"?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int){self.sample_rate},+format=(string)S16LE,+channels=(int)1"
        
        # Full URLs
        self.vi_asr_url = f"{vi_asr_url}/voice/api/asr/v1/ws/decode_online{self.content_type_param}"
        self.en_asr_url = f"{en_asr_url}/voice/api/asr/v1/ws/decode_online{self.content_type_param}"
        
        # Audio processing setup
        self.audio_process = None
        self.audio_queue = []
        self.is_running = True
        
        # WebSocket connections
        self.vi_ws = None
        self.en_ws = None
        
        # Track number of messages
        self.vi_messages_received = 0
        self.en_messages_received = 0
        
        # Reconnection flags
        self.reconnect_vi = False
        self.reconnect_en = False
        
        # Record when audio starts
        self.recording_start_time = None
        
        # Connect to both servers
        if self.connect():
            # Initialize audio streaming
            if self.test_mode:
                self.start_test_audio()
            elif self.initialize_audio_stream():
                self.start_recording()
            else:
                print("Failed to initialize audio recording.", file=sys.stderr)
        else:
            print("Failed to connect to one or both ASR servers.", file=sys.stderr)
    
    def start_test_audio(self):
        """Generate test audio (sine wave) to test ASR services"""
        print("Starting test audio generation...")
        self.recording_start_time = get_timestamp()
        
        # Create test audio that's likely to trigger speech recognition
        duration = 3  # seconds
        volume = 32767  # max volume for 16-bit audio
        
        # Generate time array
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)
        
        # Start with silence
        audio = np.zeros(len(t), dtype=np.int16)
        
        # Add a simple spoken word pattern (simulate "hello")
        # First syllable ("he")
        start = int(0.5 * self.sample_rate)
        end = int(1.0 * self.sample_rate)
        audio[start:end] = (volume * 0.7 * np.sin(2 * np.pi * 200 * t[0:end-start])).astype(np.int16)
        
        # Second syllable ("lo")
        start = int(1.1 * self.sample_rate)
        end = int(1.7 * self.sample_rate)
        audio[start:end] = (volume * 0.8 * np.sin(2 * np.pi * 150 * t[0:end-start])).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        
        # Split into chunks and add to queue
        for i in range(0, len(audio_bytes), self.buffer_size):
            chunk = audio_bytes[i:i+self.buffer_size]
            if len(chunk) < self.buffer_size:
                # Pad with zeros if needed
                chunk = chunk + bytes(self.buffer_size - len(chunk))
            self.audio_queue.append(chunk)
        
        print(f"Added {len(self.audio_queue)} test audio chunks to queue")
        print("Speaking the sample word 'hello'")
        
    def initialize_audio_stream(self):
        """Initialize audio recording with arecord"""
        cmd = ["arecord", "-D", "default", "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1", "-t", "raw"]
        try:
            self.audio_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Started arecord process with command: {' '.join(cmd)}")
            return True
        except Exception as e:
            print(f"Error starting arecord: {e}")
            return False
    
    def connect(self):
        """Connect to both ASR servers"""
        vi_running = self.is_server_running(self.vi_base_url)
        en_running = self.is_server_running(self.en_base_url)
        
        if not vi_running:
            print("Vietnamese ASR server is not running.", file=sys.stderr)
        if not en_running:
            print("English ASR server is not running.", file=sys.stderr)
        
        if not (vi_running or en_running):
            return False
            
        try:
            # Connect to Vietnamese ASR
            if vi_running:
                self.connect_vi_ws()
            
            # Connect to English ASR  
            if en_running:
                self.connect_en_ws()
                
            return True
        except Exception as e:
            print(f"Error while connecting to the servers: {e}")
            return False
    
    def connect_vi_ws(self):
        """Connect to Vietnamese ASR WebSocket"""
        if hasattr(self, 'vi_ws') and self.vi_ws:
            try:
                self.vi_ws.close()
            except:
                pass
        
        self.vi_ws = WebSocketApp(
            self.vi_asr_url,
            on_message=self.on_vi_message,
            on_error=lambda ws, err: self.on_vi_error(ws, err),
            on_close=lambda ws, code, msg: self.on_vi_close(ws, code, msg),
            on_open=self.on_vi_open
        )
        self.vi_ws_thread = threading.Thread(target=self.vi_ws.run_forever)
        self.vi_ws_thread.daemon = True
        self.vi_ws_thread.start()
        print("Vietnamese WebSocket connection established.")
        debug_print(f"Vietnamese WebSocket URL: {self.vi_asr_url}")
    
    def connect_en_ws(self):
        """Connect to English ASR WebSocket"""
        if hasattr(self, 'en_ws') and self.en_ws:
            try:
                self.en_ws.close()
            except:
                pass
                
        self.en_ws = WebSocketApp(
            self.en_asr_url,
            on_message=self.on_en_message,
            on_error=lambda ws, err: self.on_en_error(ws, err),
            on_close=lambda ws, code, msg: self.on_en_close(ws, code, msg),
            on_open=self.on_en_open
        )
        self.en_ws_thread = threading.Thread(target=self.en_ws.run_forever)
        self.en_ws_thread.daemon = True
        self.en_ws_thread.start()
        print("English WebSocket connection established.")
        debug_print(f"English WebSocket URL: {self.en_asr_url}")
    
    def is_server_running(self, url):
        """Check if server is running at given URL"""
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        port = parsed_url.port or 80
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex((host, port)) == 0
            debug_print(f"Server at {host}:{port} is {'running' if result else 'not running'}")
            return result
    
    def start_recording(self):
        """Start recording and processing audio"""
        self.recording_thread = threading.Thread(target=self.record_and_send_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        self.recording_start_time = get_timestamp()
        print(f"Recording started at {self.recording_start_time}")
        print("Speak now - results will appear below...")
    
    def record_and_send_audio(self):
        """Record audio data using arecord and stream it over WebSockets"""
        try:
            print("Recording and sending audio using arecord...")
            while self.is_running:
                chunk = self.audio_process.stdout.read(self.buffer_size)
                if not chunk:
                    break
                # Add chunk to queue for both services
                self.audio_queue.append(chunk)
                # Print a dot every 10 chunks to show activity
                if len(self.audio_queue) % 10 == 0:
                    debug_print(f"Audio queue size: {len(self.audio_queue)}")
        except Exception as e:
            print(f"Error sending audio data: {e}")
        finally:
            self.cleanup_audio()
            self.is_running = False
    
    def cleanup_audio(self):
        """Clean up audio resources"""
        if self.audio_process:
            self.audio_process.terminate()
            self.audio_process = None
    
    def get_confidence_score(self, hypothesis):
        """Extract confidence score from hypothesis"""
        try:
            if isinstance(hypothesis, dict):
                if "confidence" in hypothesis:
                    return float(hypothesis["confidence"])
                elif "score" in hypothesis:
                    return float(hypothesis["score"])
                elif "word_alignment" in hypothesis and hypothesis["word_alignment"]:
                    confidences = [word.get("confidence", 0) for word in hypothesis["word_alignment"] if "confidence" in word]
                    return sum(confidences) / len(confidences) if confidences else 0
            return 0.0
        except Exception as e:
            debug_print(f"Error getting confidence score: {e}")
            return 0.0
    
    def process_asr_result(self, language, data):
        """Process ASR result from either language"""
        timestamp = get_timestamp()
        
        try:
            debug_print(f"Processing {language} ASR result: {data}")
            
            if 'result' not in data:
                debug_print(f"No 'result' field in data: {data}")
                return
                
            is_final = data['result'].get('final', False)
            
            if 'hypotheses' not in data['result'] or not data['result']['hypotheses']:
                debug_print(f"No hypotheses in result: {data['result']}")
                return
                
            hypothesis = data["result"]["hypotheses"][0]
            
            # Get transcript (normalized for final, regular for interim)
            if is_final:
                transcript = hypothesis.get("transcript_normalized", hypothesis.get("transcript", ""))
            else:
                transcript = hypothesis.get("transcript", "")
                
            # Extract confidence score
            confidence = self.get_confidence_score(hypothesis)
            
            # Log result
            self.log_result(language, timestamp, transcript, is_final, confidence)
        except Exception as e:
            print(f"Error processing ASR result: {e}")
    
    def on_vi_message(self, ws, message):
        """Handle Vietnamese ASR message"""
        self.vi_messages_received += 1
        debug_print(f"Received Vietnamese message #{self.vi_messages_received}: {message[:100]}...")
        try:
            data = json.loads(message)
            self.process_asr_result('vi', data)
        except json.JSONDecodeError as e:
            print(f"Error decoding Vietnamese message: {e}")
            debug_print(f"Message content: {message}")
    
    def on_en_message(self, ws, message):
        """Handle English ASR message"""
        self.en_messages_received += 1
        debug_print(f"Received English message #{self.en_messages_received}: {message[:100]}...")
        try:
            data = json.loads(message)
            self.process_asr_result('en', data)
        except json.JSONDecodeError as e:
            print(f"Error decoding English message: {e}")
            debug_print(f"Message content: {message}")
    
    def on_vi_error(self, ws, error):
        """Handle Vietnamese ASR WebSocket error"""
        print(f"Vietnamese WebSocket error: {error}")
        self.reconnect_vi = True
    
    def on_en_error(self, ws, error):
        """Handle English ASR WebSocket error"""
        print(f"English WebSocket error: {error}")
        self.reconnect_en = True
    
    def on_vi_close(self, ws, code, message):
        """Handle Vietnamese ASR WebSocket close"""
        print(f"Vietnamese ASR connection closed: {code} - {message}")
        self.reconnect_vi = True
    
    def on_en_close(self, ws, code, message):
        """Handle English ASR WebSocket close"""
        print(f"English ASR connection closed: {code} - {message}")
        self.reconnect_en = True
    
    def on_vi_open(self, ws):
        """Handle Vietnamese ASR WebSocket open"""
        print("Vietnamese WebSocket opened, starting audio thread...")
        self.start_audio_sending_thread(ws, 'vi')
    
    def on_en_open(self, ws):
        """Handle English ASR WebSocket open"""
        print("English WebSocket opened, starting audio thread...")
        self.start_audio_sending_thread(ws, 'en')
    
    def start_audio_sending_thread(self, ws, lang):
        """Start thread to send audio chunks to WebSocket"""
        def run():
            last_sent_index = 0
            while self.is_running:
                try:
                    # Check if there are new chunks to send
                    if len(self.audio_queue) > last_sent_index:
                        chunk = self.audio_queue[last_sent_index]
                        try:
                            ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
                            debug_print(f"Sent audio chunk #{last_sent_index} to {lang} ASR")
                        except Exception as e:
                            debug_print(f"Error sending audio to {lang} ASR: {e}")
                            break  # Exit the thread if we can't send
                        
                        last_sent_index += 1
                        # Sleep a bit to not overwhelm the server
                        time.sleep(0.1)  # Slower sending rate to avoid overwhelming
                    else:
                        # No new chunks, sleep a bit
                        time.sleep(0.1)
                except Exception as e:
                    debug_print(f"Error in audio sending thread for {lang}: {e}")
                    break
                
        # Start the sending thread
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def log_result(self, lang, timestamp, text, is_final, confidence):
        """Log and display ASR results"""
        result_type = "FINAL" if is_final else "INTERIM"
        
        # Display the result in the console with colors
        if lang == 'vi':
            color_code = '\033[92m'  # Green for Vietnamese
        else:
            color_code = '\033[93m'  # Yellow for English
            
        reset_code = '\033[0m'
        
        # Display all results - don't filter by confidence
        print(f"{color_code}[{timestamp}] [{lang.upper()}] [{result_type}] [Confidence: {confidence:.2f}] {text}{reset_code}")

def main():
    """Entry point for the application"""
    # Check for test mode
    test_mode = "--test" in sys.argv
    
    # Create the simple ASR client
    asr_client = SimpleAsrClient(test_mode=test_mode)
    
    try:
        # Keep the main thread running
        while asr_client.is_running:
            time.sleep(0.1)
            
            # Handle reconnections if needed
            if asr_client.reconnect_vi:
                print("Attempting to reconnect to Vietnamese ASR...")
                asr_client.connect_vi_ws()
                asr_client.reconnect_vi = False
                
            if asr_client.reconnect_en:
                print("Attempting to reconnect to English ASR...")
                asr_client.connect_en_ws()
                asr_client.reconnect_en = False
            
            # Print status updates occasionally
            if DEBUG and time.time() % 5 < 0.1:
                debug_print(f"VI messages: {asr_client.vi_messages_received}, EN messages: {asr_client.en_messages_received}")
                
    except KeyboardInterrupt:
        print("\nStopping the application...")
    finally:
        # Clean up resources
        asr_client.is_running = False

if __name__ == "__main__":
    main() 