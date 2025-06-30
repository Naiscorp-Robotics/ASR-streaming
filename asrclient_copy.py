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
from scipy.signal import resample
import pyaudio
from pydub import AudioSegment as auseg
import dataclasses

DEFAULT_ASR_URL = "ws://localhost:9432"
BUFFER_SIZE = 8000
SAMPLE_RATE = 8000

def get_highest_sample_rate(audio_interface, device_index):
    """Get the highest supported sample rate for the specified device."""
    try:
        device_info = audio_interface.get_device_info_by_index(device_index)
        max_rate = int(device_info['defaultSampleRate'])
        
        if 'supportedSampleRates' in device_info:
            supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
            print(f"Device {device_index} has supported rates: {supported_rates}")
            if supported_rates:
                max_rate = max(supported_rates)
        
        return max_rate
    except Exception as e:
        print(f"Failed to get highest sample rate: {e}")
        return 48000

def validate_device(audio_interface, device_index):
    try:
        device_info = audio_interface.get_device_info_by_index(device_index)
        if not device_info.get('maxInputChannels', 0) > 0:
            return False

        # Try to actually read from the device
        test_stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=2048,
            input_device_index=device_index,
            start=False 
        )

        # Start the stream and try to read from it
        test_stream.start_stream()
        test_data = test_stream.read(2048, exception_on_overflow=False)
        test_stream.stop_stream()
        test_stream.close()

        if len(test_data) == 0:
            return False

        return True

    except Exception as e:
        print(f"Device validation failed: {e}")
        return False
    
def clear_line():
    print('\r\033[K', end="", flush=True)

def write(text):
    clear_line()
    print(text, end="", flush=True)

class AsrStreamingRecoderClient:
    def __init__(self,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 asr_url: str = DEFAULT_ASR_URL,
                 output_wav_file: str = None,
                 auto_start: bool = True
                 ):
   
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.asr_url         = f"{asr_url}/voice/api/asr/v1/ws/decode_online?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int){self.sample_rate}"
        self.wav_file        = None
        self.output_wav_file = output_wav_file

        # device 
        self.stream = None
        self.input_device_index = None
        self.shutdown_event = mp.Event()
        self.audio_queue    = mp.Queue()
        self.is_running = True
        self.ws = None
        self.ws_thread = None

        self.realtime_text = ""
        self.final_text    = ""

        # Only start recording if auto_start is True
        if auto_start:
            if not self.connect():
                print("Failed to connect to the server.", file=sys.stderr)
            else:
                self.is_init_audio = self.initialize_audio_stream()
                self.start_recording()

    def initialize_audio_stream(self):
        """Initialize the audio stream with error handling."""
        num_try = 0
        self.audio_interface = pyaudio.PyAudio()
        while not self.shutdown_event.is_set():
            try:
                num_try += 1
                # First, get a list of all available input devices
                input_devices = []
                for i in range(self.audio_interface.get_device_count()):
                    try:
                        device_info = self.audio_interface.get_device_info_by_index(i)
                        if device_info.get('maxInputChannels', 0) > 0:
                            input_devices.append(i)
                    except Exception:
                        continue

                if not input_devices:
                    raise Exception("No input devices found")

                # If input_device_index is None or invalid, try to find a working device
                if self.input_device_index is None or self.input_device_index not in input_devices:
                    # First try the default device
                    try:
                        default_device = self.audio_interface.get_default_input_device_info()
                        if validate_device(self.audio_interface, default_device['index']):
                            self.input_device_index = default_device['index']
                    except Exception:
                        # If default device fails, try other available input devices
                        for device_index in input_devices:
                            if validate_device(self.audio_interface, device_index):
                                self.input_device_index = device_index
                                break
                        else:
                            raise Exception("No working input devices found")
                self.device_sample_rate = get_highest_sample_rate(self.audio_interface, self.input_device_index)
                        
                # If we get here, we have a validated device
                self.stream = self.audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.device_sample_rate,
                    input=True,
                    frames_per_buffer=self.buffer_size,
                    input_device_index=self.input_device_index,
                )

                print(f"Microphone connected with input_device_index: {self.input_device_index} and sample_rate: {self.device_sample_rate}")

                return True

            except Exception as e:
                if num_try >= 3: break
                print(f"Microphone connection failed: {e}. Retrying...", exc_info=True)
                self.input_device_index = None
                time.sleep(3)
                continue

        print(f"Error initializing pyaudio audio recording")
        if self.audio_interface:
            self.audio_interface.terminate()
        return False
   

    def preprocess_audio(self, chunk, orin_sr, tg_sr):
        """Preprocess audio chunk similar to feed_audio method."""
        if isinstance(chunk, np.ndarray):
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)
            if orin_sr != tg_sr:
                num_samples = int(len(chunk) * tg_sr / orin_sr)
                chunk = resample(chunk, num_samples)
        else:
            if orin_sr != tg_sr:
                chunk = auseg(data=chunk,
                    sample_width=2,
                    frame_rate=self.device_sample_rate,
                    channels=1
                ).set_frame_rate(self.sample_rate)._data
        return chunk

    def connect(self):
        if not self.is_server_running():
            print("ASR Streaming server is not running.", file=sys.stderr)
            return False
        try:
            self.ws = WebSocketApp(self.asr_url,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close,
                                    on_open=self.on_open)

            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = False
            self.ws_thread.start()

            print("WebSocket connections established successfully.")
            return True
        
        except Exception as e:
            print(f"Error while connecting to the server: {e}")
            return False

    def is_server_running(self):
        parsed_url = urlparse(self.asr_url)
        host = parsed_url.hostname
        port = parsed_url.port or 80
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0


    def start_recording(self):
        self.recording_thread = threading.Thread(target=self.record_and_send_audio)
        self.recording_thread.daemon = False
        self.recording_thread.start()

    def record_and_send_audio(self):
        """Record and stream audio data"""
        try:
            if not self.is_init_audio:
                raise Exception("Failed to set up audio recording.")

            if self.output_wav_file and not self.wav_file:
                self.wav_file = wave.open(self.output_wav_file, 'wb')
                self.wav_file.setnchannels(1)
                self.wav_file.setsampwidth(2)
                self.wav_file.setframerate(self.device_sample_rate) 

            print("Recording and sending audio...", end="", flush=True)
            while self.is_running:
                try:
                    audio_data = self.stream.read(self.buffer_size, exception_on_overflow=False)
                    if self.wav_file:
                        self.wav_file.writeframes(audio_data)
                    audio_data = self.preprocess_audio(audio_data, self.device_sample_rate, self.sample_rate)
                    self.audio_queue.put_nowait(audio_data)
                except KeyboardInterrupt:
                    print("KeyboardInterrupt in record_and_send_audio, exiting...")
                    break
                except Exception as e:
                    print(f"Error sending audio data: {e}")
                    break

        except Exception as e:
            print(f"Error in record_and_send_audio: {e}", file=sys.stderr)
        finally:
            self.cleanup_audio()
            self.is_running = False

    def cleanup_audio(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None

    def on_message(self, ws, message):
        data = json.loads(message)
        if not 'result' in data:
            return None
        elif data['result']['final']:
            transcript = data["result"]["hypotheses"][0]["transcript_normalized"]
            self.final_text += transcript[0].upper() + transcript[1:] + ". "
            try:
                is_speaker = data["is_speaker"]
            except:
                is_speaker = True
            if is_speaker:
                write('\033[92m' + self.final_text)
            else:
                write('\033[91m' + self.final_text)
        else:
            transcript = data["result"]["hypotheses"][0]["transcript"]
            if transcript != self.realtime_text:
                self.realtime_text = transcript
                write('\033[93m' + self.realtime_text)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.is_running = False

    def on_open(self, ws):
        def run():
            while True:
                if self.audio_queue.qsize() != 0:
                    chunk = self.audio_queue.get()
                    ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
                else:
                    time.sleep(0.01)
                if not self.is_running:
                    break
        my_thread = threading.Thread(target=run)
        my_thread.start()

    def shutdown(self):
        """Shutdown all resources"""
        self.is_running = False
        self.shutdown_event.set()
        if self.ws:
            self.ws.close()

        # Join threads
        if self.ws_thread:
            self.ws_thread.join()
        if self.recording_thread:
            self.recording_thread.join()

        # Clean up audio
        self.cleanup_audio()

    def __enter__(self):
        """
        Method to setup the context manager protocol.

        This enables the instance to be used in a `with` statement, ensuring
        proper resource management. When the `with` block is entered, this
        method is automatically called.

        Returns:
            self: The current instance of the class.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Method to define behavior when the context manager protocol exits.

        This is called when exiting the `with` block and ensures that any
        necessary cleanup or resource release processes are executed, such as
        shutting down the system properly.

        Args:
            exc_type (Exception or None): The type of the exception that
              caused the context to be exited, if any.
            exc_value (Exception or None): The exception instance that caused
              the context to be exited, if any.
            traceback (Traceback or None): The traceback corresponding to the
              exception, if any.
        """
        self.shutdown()

    def process_wav_file(self, input_wav_path):
        """
        Process an existing WAV file and get its transcript.
        
        Args:
            input_wav_path: Path to the WAV file to transcribe
            
        Returns:
            The final transcript text
        """
        # Connect only if no active connection
        if not self.ws or not self.ws_thread or not self.ws_thread.is_alive():
            if not self.connect():
                print("Failed to connect to the server.", file=sys.stderr)
                return None
            
        try:
            # Reset the text buffers
            self.realtime_text = ""
            self.final_text = ""
            
            print(f"Processing WAV file: {input_wav_path}")
            
            # Open the WAV file
            with wave.open(input_wav_path, 'rb') as wav_file:
                # Get file properties
                file_sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                
                # Read data in chunks and send to the server
                chunk_size = self.buffer_size
                
                # Create a threading Event to signal when processing is complete
                done_event = threading.Event()
                
                # Override the on_close method temporarily to set the done event
                original_on_close = self.on_close
                def temp_on_close(ws, close_status_code, close_msg):
                    original_on_close(ws, close_status_code, close_msg)
                    done_event.set()
                self.on_close = temp_on_close
                
                # Start sending the audio data
                while True:
                    audio_data = wav_file.readframes(chunk_size)
                    if not audio_data:
                        break
                    
                    # If stereo, convert to mono
                    if n_channels == 2:
                        # Convert bytes to numpy array
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        # Reshape to (n_samples, 2)
                        audio_array = audio_array.reshape(-1, 2)
                        # Average the channels
                        audio_array = np.mean(audio_array, axis=1, dtype=np.int16)
                        # Convert back to bytes
                        audio_data = audio_array.tobytes()
                    
                    # Resample if needed
                    if file_sample_rate != self.sample_rate:
                        audio_data = self.preprocess_audio(audio_data, file_sample_rate, self.sample_rate)
                    
                    # Check if WebSocket is still connected
                    if not self.ws:
                        print("WebSocket connection lost")
                        break
                        
                    # Send the audio data
                    try:
                        self.ws.send(audio_data, opcode=ABNF.OPCODE_BINARY)
                        time.sleep(0.01)  # Small delay to avoid flooding the server
                    except Exception as e:
                        print(f"Error sending audio: {e}")
                        break
                
                # Wait for some time to allow final processing
                time.sleep(1.0)
                
                # Send a close signal to the server
                if self.ws:
                    try:
                        self.ws.close()
                    except:
                        pass
                
                # Wait for processing to complete or timeout
                done_event.wait(timeout=10.0)
                
                # Restore the original on_close method
                self.on_close = original_on_close
                
                print("\nTranscription completed!")
                return self.final_text
                
        except Exception as e:
            print(f"Error processing WAV file: {e}")
            return None
        finally:
            self.shutdown()

def transcribe_wav_file(input_wav_path, asr_url=DEFAULT_ASR_URL):
    """
    Helper function to transcribe a WAV file without recording audio.
    
    Args:
        input_wav_path: Path to the WAV file to transcribe
        asr_url: URL of the ASR server
        
    Returns:
        The transcript text
    """
    # Create client without auto-starting recording
    client = AsrStreamingRecoderClient(asr_url=asr_url, auto_start=False)
    
    # Process the file
    transcript = client.process_wav_file(input_wav_path)
    return transcript

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ASR Streaming Client')
    parser.add_argument('--wav', type=str, help='Path to WAV file for transcription')
    parser.add_argument('--output', type=str, default=None, help='Path to save recorded audio')
    parser.add_argument('--url', type=str, default=DEFAULT_ASR_URL, help='ASR server URL')
    
    args = parser.parse_args()
    
    if args.wav:
        # Process existing WAV file
        transcript = transcribe_wav_file(args.wav, args.url)
        print(f"\nFinal transcript: {transcript}")
    else:
        # Start live recording mode
        streaming_trancriptor = AsrStreamingRecoderClient(output_wav_file=args.output, asr_url=args.url)