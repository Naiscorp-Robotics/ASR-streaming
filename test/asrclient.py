from urllib.parse import urlparse
import numpy as np
import threading
import multiprocessing as mp
import socket
import wave
import json
import time
import sys
import subprocess
from scipy.signal import resample
from dataclasses import dataclass
from typing import List, Optional
import re
from datetime import datetime
from collections import deque
from websocket import WebSocketApp, ABNF
from transformers import pipeline

# Constants
VI_ASR_URL = "ws://localhost:9432"
EN_ASR_URL = "ws://localhost:9433"
BUFFER_SIZE = 8000
SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = 0.7

def get_timestamp():
    """Return current timestamp as HH:MM:SS.mmm"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def clear_line():
    """Clear current console line"""
    print('\r\033[K', end="", flush=True)

@dataclass
class WordSegment:
    """Represents a single word segment with language detection"""
    text: str
    start_time: float
    end_time: float
    confidence: float
    language: str
    is_final: bool
    timestamp: str

    def duration(self):
        return self.end_time - self.start_time
    
    def __str__(self):
        return f"[{self.language}][{self.start_time:.2f}-{self.end_time:.2f}] '{self.text}' ({self.confidence:.2f})"


class BilingualASRMerger:
    """Merges English and Vietnamese ASR outputs with language detection and conflict resolution"""
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 context_window_size: int = 5,
                 max_time_diff_ms: int = 200,
                 dominant_language: str = 'vi',
                 enable_logging: bool = True):
        
        self.confidence_threshold = confidence_threshold
        self.context_window_size = context_window_size
        self.max_time_diff_ms = max_time_diff_ms / 1000.0
        self.dominant_language = dominant_language
        self.enable_logging = enable_logging
        
        # Word segments storage
        self.vi_segments: List[WordSegment] = []
        self.en_segments: List[WordSegment] = []
        self.merged_segments: List[WordSegment] = []
        
        # Output texts
        self.merged_text = ""
        self.merged_text_by_lang = {"vi_raw": "", "vi_corrected": "", "en": ""}
        
        # Context tracking
        self.recent_language = dominant_language
        self.language_switches = 0
        
        # Confidence score tracking
        self.vi_confidence_sum = 0.0
        self.en_confidence_sum = 0.0
        self.vi_segment_count = 0
        self.en_segment_count = 0
        
        # Simple dictionaries for word validation
        self.vi_dict = self._load_vi_dictionary()
        self.en_dict = self._load_en_dictionary()
        
        # Timeline buffer for alignment
        self.timeline_buffer = deque(maxlen=self.context_window_size * 2)
        
        # Record start time
        self.start_time = time.time()
        self.log_file = None
        
        # Initialize Vietnamese corrector
        try:
            self.vi_corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
            self.log_info("Initialized Vietnamese text corrector model.")
        except Exception as e:
            self.vi_corrector = None
            self.log_info(f"Failed to initialize Vietnamese text corrector: {e}")
        
        if enable_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(f"bilingual_merger_{timestamp}.log", "w", encoding="utf-8")
            self.log_info("Bilingual ASR Merger initialized")
    
    def _load_vi_dictionary(self):
        """Load Vietnamese dictionary (simplified demo version)"""
        # In production, load from a file
        return {
            "xin", "chào", "cảm", "ơn", "tạm", "biệt", "tôi", "bạn", 
            "và", "hoặc", "người", "thời", "gian", "ngày", "đêm"
        }
    
    def _load_en_dictionary(self):
        """Load English dictionary (simplified demo version)"""
        # In production, load from a file
        return {
            "hello", "thank", "you", "goodbye", "i", "me", "my", "mine",
            "and", "or", "the", "time", "day", "night", "person"
        }
    
    def _is_vietnamese_text(self, text: str) -> bool:
        """Check if text contains Vietnamese-specific characters"""
        vi_pattern = re.compile(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', re.IGNORECASE)
        return bool(vi_pattern.search(text))
    
    def _detect_language(self, text: str) -> str:
        """Detect if a word is more likely English or Vietnamese"""
        word = text.lower().strip()
        
        if self._is_vietnamese_text(word):
            return 'vi'
        
        in_vi_dict = word in self.vi_dict
        in_en_dict = word in self.en_dict
        
        if in_vi_dict and not in_en_dict:
            return 'vi'
        elif in_en_dict and not in_vi_dict:
            return 'en'
        
        return self.recent_language
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert HH:MM:SS.mmm timestamp to seconds"""
        try:
            time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
        except ValueError:
            return time.time() - self.start_time
    
    def _calculate_overlap(self, seg1: WordSegment, seg2: WordSegment) -> float:
        """Calculate time overlap between two segments in seconds"""
        overlap_start = max(seg1.start_time, seg2.start_time)
        overlap_end = min(seg1.end_time, seg2.end_time)
        
        return max(0.0, overlap_end - overlap_start)
    
    def _has_timestamp_conflict(self, seg1: WordSegment, seg2: WordSegment) -> bool:
        """Check if two segments have a timestamp conflict"""
        overlap = self._calculate_overlap(seg1, seg2)
        min_duration = min(seg1.duration(), seg2.duration())
        
        return overlap > (min_duration / 2)
    
    def _resolve_conflict(self, vi_seg: WordSegment, en_seg: WordSegment) -> WordSegment:
        """Resolve conflict between Vietnamese and English segments"""
        self.log_info(f"Resolving conflict between: {vi_seg} and {en_seg}")
        
        # 1. Compare confidence scores
        if abs(vi_seg.confidence - en_seg.confidence) > 0.1:
            if vi_seg.confidence > en_seg.confidence:
                self.log_info("Selected VI segment due to higher confidence")
                return vi_seg
            else:
                self.log_info("Selected EN segment due to higher confidence")
                return en_seg
        
        # 2. Apply lexical validity check
        vi_valid = vi_seg.text.lower() in self.vi_dict or self._is_vietnamese_text(vi_seg.text)
        en_valid = en_seg.text.lower() in self.en_dict
        
        if vi_valid and not en_valid:
            self.log_info("Selected VI segment due to lexical validity")
            return vi_seg
        elif en_valid and not vi_valid:
            self.log_info("Selected EN segment due to lexical validity")
            return en_seg
        
        # 3. Check context - prefer to continue in same language
        if self.recent_language == 'vi':
            self.log_info("Selected VI segment due to context")
            return vi_seg
        elif self.recent_language == 'en':
            self.log_info("Selected EN segment due to context")
            return en_seg
        
        # 4. Default to dominant language
        self.log_info(f"Selected {self.dominant_language} segment as default")
        return vi_seg if self.dominant_language == 'vi' else en_seg
    
    def add_result(self, language: str, timestamp: str, text: str, is_final: bool, confidence: float):
        """Add a new ASR result from either language"""
        if language not in ['vi', 'en']:
            self.log_info(f"Invalid language code: {language}")
            return
            
        # Convert timestamp to seconds from start
        time_seconds = self._timestamp_to_seconds(timestamp)
        
        # Create segment with estimated duration
        estimated_duration = len(text.split()) * 0.3  # ~300ms per word
        
        segment = WordSegment(
            text=text,
            start_time=time_seconds,
            end_time=time_seconds + estimated_duration,
            confidence=confidence,
            language=language,
            is_final=is_final,
            timestamp=timestamp
        )
        
        # Add to the appropriate language segment list
        if language == 'vi':
            self.vi_segments.append(segment)
            if is_final:
                self.vi_confidence_sum += confidence
                self.vi_segment_count += 1
                self.log_info(f"Added VI confidence: {confidence}, new total: {self.vi_confidence_sum}, count: {self.vi_segment_count}")
        else:
            self.en_segments.append(segment)
            if is_final:
                self.en_confidence_sum += confidence
                self.en_segment_count += 1
                self.log_info(f"Added EN confidence: {confidence}, new total: {self.en_confidence_sum}, count: {self.en_segment_count}")
            
        self.log_info(f"Added {language.upper()} segment: {segment}")
        
        # Process immediately if final
        if is_final:
            self._process_new_segments()
    
    def _process_new_segments(self):
        """Process all accumulated segments and merge them"""
        vi_final = [seg for seg in self.vi_segments if seg.is_final and seg not in self.merged_segments]
        en_final = [seg for seg in self.en_segments if seg.is_final and seg not in self.merged_segments]
        
        # Sort by timestamp
        all_segments = sorted(vi_final + en_final, key=lambda x: x.start_time)
        
        last_segment = None
        for segment in all_segments:
            if last_segment and self._is_duplicate(last_segment, segment):
                continue  # Skip duplicate
            
            if segment.language == 'vi':
                conflicts = [
                    seg for seg in en_final 
                    if self._has_timestamp_conflict(segment, seg) and seg not in self.merged_segments
                ]
            else:
                conflicts = [
                    seg for seg in vi_final 
                    if self._has_timestamp_conflict(segment, seg) and seg not in self.merged_segments
                ]
            
            if conflicts:
                conflicts.sort(key=lambda x: abs(x.start_time - segment.start_time))
                closest_conflict = conflicts[0]
                if segment.language == 'vi':
                    winning_segment = self._resolve_conflict(segment, closest_conflict)
                else:
                    winning_segment = self._resolve_conflict(closest_conflict, segment)
                self._add_to_merged_output(winning_segment)
            else:
                self._add_to_merged_output(segment)

            last_segment = segment  # Keep track of the last added segment

        self._update_merged_text()
        
    def _is_duplicate(self, last_segment, current_segment):
        """Check if two consecutive segments are duplicates based on text and timestamps"""
        # Consider segments duplicates if they have identical text and are close in time
        return (last_segment.text.lower() == current_segment.text.lower() and
                abs(last_segment.start_time - current_segment.start_time) < 0.2)  # Adjust threshold if needed

    
    def _add_to_merged_output(self, segment: WordSegment):
        """Add a segment to the merged output"""
        if segment not in self.merged_segments:
            self.merged_segments.append(segment)
            
            # Update language context
            if segment.language != self.recent_language:
                self.language_switches += 1
                self.recent_language = segment.language
                self.log_info(f"Language switch to {segment.language}, total switches: {self.language_switches}")
    
    def _update_merged_text(self):
        """Update the merged text output based on merged segments"""
        # Sort merged segments by start time
        sorted_segments = sorted(self.merged_segments, key=lambda x: x.start_time)
        
        # Create merged text with language tags and raw language-specific texts
        merged_text = ""
        raw_vi_text = ""
        en_text = ""
        current_language = None
        
        for segment in sorted_segments:
            # Skip low confidence segments from combined output
            if segment.confidence > self.confidence_threshold: 
                continue
                
            # Add language marker on language switch in merged text
            if segment.language != current_language:
                if merged_text:
                    merged_text += f" [{segment.language.upper()}] "
                else:
                    merged_text += f"[{segment.language.upper()}] "
                current_language = segment.language
            else:
                merged_text += " "
            
            # Add the text to merged output
            merged_text += segment.text
            
            # Add to raw language-specific outputs
            if segment.language == 'vi':
                raw_vi_text += " " + segment.text if raw_vi_text else segment.text
            else:
                en_text += " " + segment.text if en_text else segment.text
        
        # Apply Vietnamese correction to the entire accumulated raw text
        corrected_vi_text = raw_vi_text
        if self.vi_corrector and raw_vi_text:
            self.log_info(f"Attempting VI correction on accumulated text: '{raw_vi_text}'")
            try:
                # Pass the full raw_vi_text to the corrector
                correction_result = self.vi_corrector(raw_vi_text, max_length=512) 
                if correction_result and isinstance(correction_result, list) and 'generated_text' in correction_result[0]:
                    corrected_vi_text = correction_result[0]['generated_text'].strip() # Use the corrected text
                    self.log_info(f"Applied VI correction: -> '{corrected_vi_text}'")
                else:
                    self.log_info(f"VI correction returned unexpected result: {correction_result}")
            except Exception as e:
                self.log_info(f"Error during VI text correction: {e}")
        else:
             if not self.vi_corrector:
                 self.log_info("VI corrector not available, skipping correction.")
             if not raw_vi_text:
                 self.log_info("No VI text to correct.")

        # Store the final texts
        self.merged_text = merged_text
        self.merged_text_by_lang = {
            "vi_raw": raw_vi_text,        # Store raw VI text
            "vi_corrected": corrected_vi_text, # Use corrected text here
            "en": en_text
        }
        
        # Update logs
        self.log_info(f"Updated merged text: {merged_text}")
        self.log_info(f"Updated VI text (raw): {raw_vi_text}")
        self.log_info(f"Updated VI text (corrected): {corrected_vi_text}")
        self.log_info(f"Updated EN text: {en_text}")
    
    def get_merged_text(self):
        """Get current merged text"""
        return self.merged_text
    
    def get_language_specific_text(self, language=None):
        """Get text for a specific language"""
        if language:
            return self.merged_text_by_lang.get(language, "")
        return self.merged_text_by_lang
    
    def log_info(self, message):
        """Log information if logging is enabled"""
        if self.enable_logging and self.log_file:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.log_file.write(f"[{timestamp}] {message}\n")
            self.log_file.flush()
    
    def close(self):
        """Close resources"""
        if self.log_file:
            self.log_file.close()


class DualAsrStreamingClient:
    def __init__(self,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 vi_asr_url: str = VI_ASR_URL,
                 en_asr_url: str = EN_ASR_URL,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 output_wav_file: Optional[str] = None):
   
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        
        # Append content-type parameters to URLs
        content_type_param = f"?content-type=audio/x-raw,+layout=(string)interleaved,+rate=(int){self.sample_rate}"
        self.vi_asr_url = f"{vi_asr_url}/voice/api/asr/v1/ws/decode_online{content_type_param}"
        self.en_asr_url = f"{en_asr_url}/voice/api/asr/v1/ws/decode_online{content_type_param}"
        
        self.wav_file = None
        self.output_wav_file = output_wav_file

        # Audio processing setup
        self.audio_process = None
        self.shutdown_event = mp.Event()
        self.audio_queue = mp.Queue()
        self.is_running = True

        # Initialize the merger for unified results
        self.merger = BilingualASRMerger(
            confidence_threshold=confidence_threshold,
            dominant_language='vi',
            enable_logging=True
        )
        
        # Record when audio starts
        self.recording_start_time = None

        # Connect to both servers
        servers_available = self.connect()
        if not servers_available:
            print("Failed to connect to one or both ASR servers.", file=sys.stderr)
        else:
            # Initialize audio streaming using arecord
            if not self.initialize_audio_stream():
                print("Failed to initialize audio recording.", file=sys.stderr)
            self.start_recording()

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

    def preprocess_audio(self, chunk, orig_sr, target_sr):
        """Preprocess audio chunk if needed (resample)"""
        if orig_sr != target_sr:
            # Resample the audio if the sample rates differ
            num_samples = int(len(chunk) * target_sr / orig_sr)
            # Convert byte string to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            resampled = resample(audio_array, num_samples)
            return resampled.astype(np.int16).tobytes()
        return chunk

    def connect(self):
        """Connect to both ASR servers"""
        vi_running = self.is_server_running(VI_ASR_URL)
        en_running = self.is_server_running(EN_ASR_URL)
        
        if not vi_running:
            print("Vietnamese ASR server is not running.", file=sys.stderr)
        if not en_running:
            print("English ASR server is not running.", file=sys.stderr)
        
        if not (vi_running or en_running):
            return False
            
        try:
            # Connect to Vietnamese ASR
            if vi_running:
                self.vi_ws = WebSocketApp(
                    self.vi_asr_url,
                    on_message=self.on_vi_message,
                    on_error=lambda ws, err: print(f"Vietnamese WebSocket error: {err}"),
                    on_close=lambda ws, code, msg: print("Vietnamese ASR connection closed"),
                    on_open=self.on_vi_open
                )
                self.vi_ws_thread = threading.Thread(target=self.vi_ws.run_forever)
                self.vi_ws_thread.daemon = False
                self.vi_ws_thread.start()
                print("Vietnamese WebSocket connection established.")
            
            # Connect to English ASR  
            if en_running:
                self.en_ws = WebSocketApp(
                    self.en_asr_url,
                    on_message=self.on_en_message,
                    on_error=lambda ws, err: print(f"English WebSocket error: {err}"),
                    on_close=lambda ws, code, msg: print("English ASR connection closed"),
                    on_open=self.on_en_open
                )
                self.en_ws_thread = threading.Thread(target=self.en_ws.run_forever)
                self.en_ws_thread.daemon = False
                self.en_ws_thread.start()
                print("English WebSocket connection established.")
                
            return True
        except Exception as e:
            print(f"Error while connecting to the servers: {e}")
            return False

    def is_server_running(self, url):
        """Check if server is running at given URL"""
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        port = parsed_url.port or 80
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def start_recording(self):
        """Start recording and processing audio"""
        self.recording_thread = threading.Thread(target=self.record_and_send_audio)
        self.recording_thread.daemon = False
        self.recording_thread.start()
        self.recording_start_time = get_timestamp()
        print(f"Recording started at {self.recording_start_time}")
        print(f"Confidence threshold set to {self.confidence_threshold} (only higher confidence results will be shown)")

    def record_and_send_audio(self):
        """Record audio data using arecord and stream it over WebSockets"""
        try:
            if self.output_wav_file and not self.wav_file:
                self.wav_file = wave.open(self.output_wav_file, 'wb')
                self.wav_file.setnchannels(1)
                self.wav_file.setsampwidth(2)
                self.wav_file.setframerate(self.sample_rate)
            print("Recording and sending audio using arecord...")
            while self.is_running:
                chunk = self.audio_process.stdout.read(self.buffer_size)
                if not chunk:
                    break
                if self.wav_file:
                    self.wav_file.writeframes(chunk)
                # Process chunk and add to queue
                processed_chunk = self.preprocess_audio(chunk, self.sample_rate, self.sample_rate)
                self.audio_queue.put_nowait(processed_chunk)
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
        if self.wav_file:
            self.wav_file.close()
            self.wav_file = None

    def get_confidence_score(self, hypothesis):
        """Extract confidence score from hypothesis"""
        if "confidence" in hypothesis:
            return float(hypothesis["confidence"])
        elif "score" in hypothesis:
            return float(hypothesis["score"])
        elif "word_alignment" in hypothesis and hypothesis["word_alignment"]:
            confidences = [word.get("confidence", 0) for word in hypothesis["word_alignment"] if "confidence" in word]
            return sum(confidences) / len(confidences) if confidences else 0
        return 0.0

    def process_asr_result(self, language, data):
        """Process ASR result from either language"""
        timestamp = get_timestamp()
        
        if 'result' not in data:
            return
            
        is_final = data['result']['final']
        hypothesis = data["result"]["hypotheses"][0]
        
        # Get transcript (normalized for final, regular for interim)
        if is_final:
            transcript = hypothesis.get("transcript_normalized", hypothesis.get("transcript", ""))
        else:
            transcript = hypothesis.get("transcript", "")
            
        # Extract confidence score
        confidence = self.get_confidence_score(hypothesis)
        
        # Skip if below confidence threshold
        if confidence > self.confidence_threshold:
            self.log_result(language, timestamp, transcript, is_final, confidence, skipped=True)
            return
            
        # Add to merger
        self.merger.add_result(language, timestamp, transcript, is_final, confidence)
        
        # Log result
        self.log_result(language, timestamp, transcript, is_final, confidence)
        
        # Display merged results
        self.display_merged_results()

    def on_vi_message(self, ws, message):
        """Handle Vietnamese ASR message"""
        data = json.loads(message)
        self.process_asr_result('vi', data)

    def on_en_message(self, ws, message):
        """Handle English ASR message"""
        data = json.loads(message)
        self.process_asr_result('en', data)

    def on_vi_open(self, ws):
        """Handle Vietnamese ASR WebSocket open"""
        self.start_audio_sending_thread(ws)

    def on_en_open(self, ws):
        """Handle English ASR WebSocket open"""
        self.start_audio_sending_thread(ws)
        
    def start_audio_sending_thread(self, ws):
        """Start thread to send audio chunks to WebSocket"""
        def run():
            while True:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    # Put it back if we're not the last consumer
                    if ws == self.vi_ws:
                        self.audio_queue.put(chunk)
                    ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
                else:
                    time.sleep(0.01)
                if not self.is_running:
                    break
        threading.Thread(target=run).start()
    
    def log_result(self, lang, timestamp, text, is_final, confidence, skipped=False):
        """Log ASR results to a file with timestamps"""
        with open(f"asr_comparison_{self.recording_start_time.replace(':', '-')}.log", "a", encoding="utf-8") as f:
            result_type = "FINAL" if is_final else "INTERIM"
            status = "SKIPPED" if skipped else "ACCEPTED"
            f.write(f"[{timestamp}] [{lang}] [{result_type}] [CONF: {confidence:.2f}] [{status}] {text}\n")
            
    def display_merged_results(self):
        """Display merged transcription results with language identification"""
        # Clear previous output
        clear_line()
        
        # Get the merged output and language-specific outputs
        merged_text = self.merger.get_merged_text()
        lang_texts = self.merger.get_language_specific_text()
        
        # Calculate average confidence scores if segments exist
        vi_avg_confidence = self.merger.vi_confidence_sum / max(1, self.merger.vi_segment_count)
        en_avg_confidence = self.merger.en_confidence_sum / max(1, self.merger.en_segment_count)
        
        # Print the combined output with language markers
        print("\n" + "="*100)
        print("\033[96mBILINGUAL MERGED OUTPUT:\033[0m")
        print("\033[97m" + merged_text + "\033[0m")
        print("-"*100) 
        
        # Print confidence scores
        print("\033[95mCONFIDENCE SCORES:\033[0m")
        print(f"\033[97mVietnamese: Sum = {self.merger.vi_confidence_sum:.4f}, Count = {self.merger.vi_segment_count}, Avg = {vi_avg_confidence:.4f}\033[0m")
        print(f"\033[97mEnglish: Sum = {self.merger.en_confidence_sum:.4f}, Count = {self.merger.en_segment_count}, Avg = {en_avg_confidence:.4f}\033[0m")
        print("-"*100)
        
        # Print language-specific outputs
        print("\033[95mVIETNAMESE ONLY TEXT (Raw):\033[0m") 
        print("\033[92m" + lang_texts.get("vi_raw", "") + "\033[0m") # Get raw VI
        print("-"*100)
        print("\033[95mVIETNAMESE ONLY TEXT (Corrected):\033[0m") 
        print("\033[92m" + lang_texts.get("vi_corrected", "") + "\033[0m") # Get corrected VI
        print("-"*100)
        print("\033[94mENGLISH ONLY TEXT:\033[0m")
        print("\033[93m" + lang_texts.get("en", "") + "\033[0m")
        print("="*100 + "\n")

def main():
    """Entry point for the application"""
    # Create the dual ASR client with merger
    dual_asr = DualAsrStreamingClient(
        output_wav_file="bilingual_recording.wav"
    )
    
    try:
        # Keep the main thread running
        while dual_asr.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping the application...")
    finally:
        # Clean up resources
        dual_asr.is_running = False
        if hasattr(dual_asr, 'merger'):
            dual_asr.merger.close()
            
if __name__ == "__main__":
    main()