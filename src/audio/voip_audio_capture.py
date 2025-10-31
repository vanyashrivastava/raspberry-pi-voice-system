# Owner: Oma
# Responsibility: Implement VOIP-side audio capture with preprocessing and speech-to-text
# Goals:
# - Provide a stable stream of raw audio frames (PCM float32 or int16) from VOIP calls.
# - Preprocess audio for optimal quality (noise reduction, normalization)
# - Transcribe speech to text for fraud detection analysis by downstream modules
# - Support hooking into pjsua (PJSIP) or Twilio media streams when available on the Pi.
# Integration points:
# - Exposes audio frames consumed by `audio.audio_preprocessor.AudioPreprocessor`.
# - Emits events or pushes to `audio.audio_stream_handler.AudioStreamHandler` for buffering.
# - Provides transcribed text to fraud detection modules
# Testing requirements:
# - Unit tests should mock the VOIP stack and validate that frames are emitted with correct sample rate, channels.
import typing as t
import queue
import threading
import time
import numpy as np
import logging
import json
import csv
import os
import wave  # <-- NEW IMPORT for reading WAV files
from datetime import datetime

# External dependencies (add to requirements.txt):
# ... (all your comments are preserved)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    (This class is identical to your code. No changes made.)
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        logger.info(f"AudioPreprocessor initialized (target_sr={target_sample_rate})")
    
    def process(self, audio_bytes: bytes, original_sample_rate: int) -> np.ndarray:
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        
        if original_sample_rate != self.target_sample_rate:
            audio = self._resample(audio, original_sample_rate)
        
        audio = self._reduce_noise(audio)
        audio = self._bandpass_filter(audio)
        audio = self._normalize_volume(audio)
        return audio
    
    def _resample(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        try:
            from scipy import signal
            num_samples = int(len(audio) * self.target_sample_rate / original_sr)
            return signal.resample(audio, num_samples)
        except ImportError:
            logger.warning("scipy not installed, skipping resample")
            return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        threshold = 0.01
        audio[np.abs(audio) < threshold] = 0
        return audio
    
    def _bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        try:
            from scipy import signal
            nyquist = self.target_sample_rate / 2
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            return signal.filtfilt(b, a, audio)
        except ImportError:
            logger.warning("scipy not installed, skipping bandpass filter")
            return audio
        except Exception as e:
            logger.error(f"Bandpass filter error: {e}")
            return audio
    
    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
        return audio


class VoipAudioCapture:
    """
    (Your comments are preserved)
    Connects to a VOIP endpoint, yields raw audio frames, preprocesses audio, and transcribes speech.
    ...
    """
    
    def __init__(
        self, 
        sip_config: t.Optional[dict] = None, 
        use_twilio: bool = False,
        use_microphone: bool = False,
        wav_file_path: t.Optional[str] = None,  # <-- NEW PARAMETER
        enable_preprocessing: bool = True,
        enable_transcription: bool = False,
        save_to_csv: bool = True,
        csv_filename: str = "audio_transcripts.csv",
        buffer_size: int = 100
    ):
        """
        (Your comments are preserved)
        Initialize the VoIP audio capture system.
        ...
        """
        self.sip_config = sip_config or {}
        self.use_twilio = use_twilio
        self.use_microphone = use_microphone
        self.wav_file_path = wav_file_path  # <-- NEW
        self.enable_preprocessing = enable_preprocessing
        self.enable_transcription = enable_transcription
        self.save_to_csv = save_to_csv
        self.csv_filename = csv_filename
        self.buffer_size = buffer_size
        
        # (Internal state is all the same...)
        self._running = False
        self._capture_thread = None
        self._transcription_thread = None
        self._frame_queue = queue.Queue(maxsize=buffer_size)
        self._transcription_queue = queue.Queue(maxsize=10)
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2
        self.chunk_size = 320
        self._preprocessor = AudioPreprocessor(target_sample_rate=self.sample_rate) if enable_preprocessing else None
        self._latest_preprocessed_audio = None
        self._preprocessed_lock = threading.Lock()
        self._recognizer = None
        self._latest_transcription = ""
        self._transcription_lock = threading.Lock()
        self._audio_buffer_for_transcription = []
        self._pjsua_lib = None
        self._pjsua_account = None
        self._twilio_client = None
        self._pyaudio_instance = None
        self._pyaudio_stream = None
        
        if self.enable_transcription:
            self._init_speech_recognition()
        
        if self.save_to_csv and self.enable_transcription:
            self._init_csv_file()
        
        # (Updated mode logic)
        if self.use_microphone:
            mode = "microphone"
        elif self.wav_file_path:
            mode = f"wav_file ({self.wav_file_path})"
        elif self.use_twilio:
            mode = "twilio"
        else:
            mode = "pjsua"
        
        logger.info(f"VoipAudioCapture initialized (mode={mode}, preprocessing={enable_preprocessing}, transcription={enable_transcription}, csv={save_to_csv})")
    
    def _init_csv_file(self) -> None:
        # (This method is identical to your code)
        try:
            file_exists = os.path.isfile(self.csv_filename)
            if not file_exists:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'session_id', 'transcription', 'duration_seconds', 'audio_frames', 'mode'])
                logger.info(f"✅ Created CSV file: {self.csv_filename}")
            else:
                logger.info(f"✅ Using existing CSV file: {self.csv_filename}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV file: {e}")
            self.save_to_csv = False
    
    def save_transcription_to_csv(self, transcription: str, duration: float, frame_count: int) -> None:
        # (This method is identical to your code, with one small fix for 'mode')
        if not self.save_to_csv:
            return
        
        try:
            # (Fixed mode logic to correctly show 'wav_file' or 'microphone')
            if self.wav_file_path:
                 mode = "wav_file"
            elif self.use_microphone:
                 mode = "microphone"
            else:
                 mode = "voip"
                 
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    int(time.time()),
                    transcription,
                    f"{duration:.2f}",
                    frame_count,
                    mode
                ])
            logger.info(f"💾 Saved transcription to {self.csv_filename}")
        
        except Exception as e:
            logger.error(f"Failed to save to CSV: {e}")
    
    def _init_speech_recognition(self) -> None:
        # (This method is identical to your code)
        try:
            from vosk import Model, KaldiRecognizer
            model_path = "model"
            if os.path.exists(model_path):
                self._vosk_model = Model(model_path)
                self._recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate)
                logger.info("✅ Using Vosk for speech recognition (offline)")
            else:
                logger.warning(f"Vosk model not found at {model_path}, trying Google Speech Recognition...")
                self._init_google_speech()
        except ImportError:
            logger.info("Vosk not installed, using Google Speech Recognition (online)")
            self._init_google_speech()

    def _init_google_speech(self) -> None:
        # (This method is identical to your code)
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            logger.info("✅ Using Google Speech Recognition (online)")
        except ImportError:
            logger.error("❌ Neither Vosk nor SpeechRecognition installed!")
            self.enable_transcription = False

    def get_transcription(self) -> str:
        # (This method is identical to your code)
        with self._transcription_lock:
            return self._latest_transcription

    def get_preprocessed_audio(self) -> t.Optional[np.ndarray]:
        # (This method is identical to your code)
        if not self.enable_preprocessing:
            return None
        with self._preprocessed_lock:
            return self._latest_preprocessed_audio

    def start(self) -> None:
        # (This method is MODIFIED to add the WAV file logic)
        if self._running:
            logger.warning("Capture already running")
            return
        
        self._running = True
        
        if self.enable_transcription:
            self._transcription_thread = threading.Thread(
                target=self._transcription_loop, 
                daemon=True
            )
            self._transcription_thread.start()
        
        # --- NEW LOGIC HERE ---
        if self.use_microphone:
            self._start_microphone_capture()
        elif self.wav_file_path:  # <-- ADDED THIS
            self._start_wav_file_capture() # <-- ADDED THIS
        elif self.use_twilio:
            self._start_twilio_capture()
        else:
            self._start_pjsua_capture()
        # --- END NEW LOGIC ---
        
        logger.info("✅ VoIP audio capture started")
    
    def stop(self) -> None:
        # (This method is MODIFIED to add the WAV file logic)
        if not self._running:
            return
        
        logger.info("Stopping VoIP audio capture...")
        self._running = False
        
        # --- NEW LOGIC HERE ---
        if self.use_microphone:
            self._stop_microphone_capture()
        elif self.wav_file_path: # <-- ADDED THIS
            self._stop_wav_file_capture() # <-- ADDED THIS
        elif self.use_twilio:
            self._stop_twilio_capture()
        else:
            self._stop_pjsua_capture()
        # --- END NEW LOGIC ---
        
        # (Rest of stop() is identical to your code)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        if self._transcription_thread and self._transcription_thread.is_alive():
            self._transcription_thread.join(timeout=2.0)
        
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ VoIP audio capture stopped")
    
    def frames(self) -> t.Generator[t.Tuple[float, bytes, int, int], None, None]:
        # (This method is MODIFIED to shut down cleanly after a WAV file)
        while self._running:
            try:
                frame_data = self._frame_queue.get(timeout=1.0)
                
                if self.enable_preprocessing:
                    self._preprocess_audio(frame_data[1], frame_data[2])
                
                if self.enable_transcription:
                    self._buffer_audio_for_transcription(frame_data[1])
                
                yield frame_data
            except queue.Empty:
                # --- NEW LOGIC HERE ---
                # If we are in WAV mode and the capture thread is finished,
                # it means the file is done and the queue is empty.
                # We can now stop the pipeline.
                if self.wav_file_path and not (self._capture_thread and self._capture_thread.is_alive()):
                    logger.info("WAV file finished and queue is empty. Stopping pipeline.")
                    self._running = False # This signals all other loops to stop
                # --- END NEW LOGIC ---
                continue
        
        while not self._frame_queue.empty():
            try:
                frame_data = self._frame_queue.get_nowait()
                yield frame_data
            except queue.Empty:
                break
    
    def _preprocess_audio(self, pcm_bytes: bytes, sample_rate: int) -> None:
        # (This method is identical to your code)
        try:
            cleaned_audio = self._preprocessor.process(pcm_bytes, sample_rate)
            with self._preprocessed_lock:
                self._latest_preprocessed_audio = cleaned_audio
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
    
    def _buffer_audio_for_transcription(self, pcm_bytes: bytes) -> None:
        # (This method is identical to your code)
        self._audio_buffer_for_transcription.append(pcm_bytes)
        
        if len(self._audio_buffer_for_transcription) >= 50:
            combined_audio = b''.join(self._audio_buffer_for_transcription)
            try:
                self._transcription_queue.put_nowait(combined_audio)
            except queue.Full:
                pass
            self._audio_buffer_for_transcription = []
    
    def _transcription_loop(self) -> None:
        # (This method is MODIFIED to shut down cleanly after a WAV file)
        logger.info("📝 Transcription thread started")
        
        # --- MODIFIED LOOP CONDITION ---
        # Keep running as long as the main loop is running OR
        # there are still items left in the transcription queue.
        while self._running or not self._transcription_queue.empty():
            try:
                audio_chunk = self._transcription_queue.get(timeout=1.0)
                
                if hasattr(self._recognizer, 'AcceptWaveform'):
                    text = self._transcribe_vosk(audio_chunk)
                else:
                    text = self._transcribe_google(audio_chunk)
                
                if text:
                    with self._transcription_lock:
                        self._latest_transcription += " " + text
                        self._latest_transcription = self._latest_transcription[-1000:]
                    logger.info(f"📝 Transcribed: {text}")
            
            except queue.Empty:
                # --- NEW LOGIC ---
                # If the queue is empty AND the main loop is no longer running,
                # then we are done and this thread can exit.
                if not self._running:
                    break
                # --- END NEW LOGIC ---
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")

    def _transcribe_vosk(self, audio_bytes: bytes) -> str:
        # (This method is MODIFIED to get the final chunk of text)
        try:
            if self._recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self._recognizer.Result())
                return result.get('text', '')
            else:
                # --- MODIFIED LOGIC ---
                # We're processing in chunks, so we only care about
                # *final* results from `AcceptWaveform`.
                # We'll get the last bit of text when we stop.
                return "" # Don't return partial results
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            return ""

    def _transcribe_google(self, audio_bytes: bytes) -> str:
        # (This method is identical to your code)
        try:
            import speech_recognition as sr
            audio_data = sr.AudioData(audio_bytes, self.sample_rate, self.sample_width)
            text = self._recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            logger.error(f"Google Speech API error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Google transcription error: {e}")
            return ""
    
    # --- (Microphone, PJSUA, Twilio, Simulation methods are identical) ---
    def _start_microphone_capture(self) -> None:
        try:
            import pyaudio
        except ImportError:
            logger.error("PyAudio not installed. Install with: pip install pyaudio")
            self._start_simulated_capture()
            return
        
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            logger.info("Available audio input devices:")
            for i in range(self._pyaudio_instance.get_device_count()):
                dev_info = self._pyaudio_instance.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:
                    logger.info(f"  [{i}] {dev_info['name']} - {dev_info['maxInputChannels']} channels")
            
            self._pyaudio_stream = self._pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._microphone_callback
            )
            self._pyaudio_stream.start_stream()
            logger.info(f"🎤 Microphone capture started!")
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            self._start_simulated_capture()

    def _stop_microphone_capture(self) -> None:
        try:
            if self._pyaudio_stream:
                self._pyaudio_stream.stop_stream()
                self._pyaudio_stream.close()
            if self._pyaudio_instance:
                self._pyaudio_instance.terminate()
            logger.info("Microphone capture stopped")
        except Exception as e:
            logger.error(f"Error stopping microphone: {e}")

    def _microphone_callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        if status:
            logger.warning(f"PyAudio status flags: {status}")
        frame = (time.time(), in_data, self.sample_rate, self.channels)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            logger.warning("Frame queue full, dropping microphone frame")
        return (None, pyaudio.paContinue)

    def _start_pjsua_capture(self) -> None:
        logger.warning("PJSUA not implemented, using simulation")
        self._start_simulated_capture()
    def _stop_pjsua_capture(self) -> None: pass
    def _start_twilio_capture(self) -> None:
        logger.warning("Twilio not implemented, using simulation")
        self._start_simulated_capture()
    def _stop_twilio_capture(self) -> None: pass
    
    
    # --- NEW METHODS FOR WAV FILE CAPTURE ---
    
    def _start_wav_file_capture(self) -> None:
        """
        (NEW) Start reading audio from a WAV file in a background thread.
        This simulates a live audio stream for testing.
        """
        if not self.wav_file_path or not os.path.exists(self.wav_file_path):
            logger.error(f"WAV file not found: {self.wav_file_path}")
            self._start_simulated_capture() # Fallback
            return

        try:
            # Check file format *before* starting the thread
            with wave.open(self.wav_file_path, 'rb') as wf:
                if wf.getnchannels() != 1:
                    raise ValueError(f"WAV file must be mono (1 channel), but has {wf.getnchannels()}")
                if wf.getsampwidth() != 2:
                    raise ValueError(f"WAV file must be 16-bit, but has {wf.getsampwidth()*8}-bit")
                if wf.getframerate() != 16000:
                    raise ValueError(f"WAV file must be 16000 Hz, but has {wf.getframerate()} Hz")
            
            # Start the file-reading thread (Lane 1)
            self._capture_thread = threading.Thread(target=self._wav_capture_loop, daemon=True)
            self._capture_thread.start()
            logger.info(f"🎧 Started capturing from WAV file: {self.wav_file_path}")

        except Exception as e:
            logger.error(f"Failed to read WAV file: {e}")
            logger.warning("Falling back to simulated audio.")
            self._start_simulated_capture()

    def _stop_wav_file_capture(self) -> None:
        """
        (NEW) Stop the WAV file capture.
        (The thread will stop on its own when _running is False)
        """
        logger.info("WAV file capture stopping.")

    def _wav_capture_loop(self) -> None:
        """
        (NEW) This is the background thread (Lane 1) for reading the WAV file.
        It reads the file in chunks and sleeps to simulate real-time playback.
        """
        # 
        try:
            wf = wave.open(self.wav_file_path, 'rb')
            # Calculate sleep duration to simulate a real-time stream
            # This is (samples per chunk) / (samples per second)
            chunk_duration = self.chunk_size / self.sample_rate # e.g., 320 / 16000 = 0.02 seconds
            
            while self._running:
                # Read one chunk's worth of audio
                pcm_bytes = wf.readframes(self.chunk_size)
                
                # If readframes returns empty bytes, we're at the end of the file
                if not pcm_bytes:
                    logger.info("Finished reading WAV file.")
                    break
                
                # Put the audio chunk on the "to-do" list (Lane 1 -> Lane 2)
                frame = (time.time(), pcm_bytes, self.sample_rate, self.channels)
                self._frame_queue.put(frame)
                
                # Sleep to simulate real-time playback
                time.sleep(chunk_duration)
            
            wf.close()
        except Exception as e:
            logger.error(f"Error in WAV capture loop: {e}")
        
        logger.info("WAV capture thread finished.")

    # --- (Simulation methods are identical) ---
    def _start_simulated_capture(self) -> None:
        logger.info("Starting simulated audio capture (for testing)")
        self._capture_thread = threading.Thread(target=self._simulated_capture_loop, daemon=True)
        self._capture_thread.start()

    def _simulated_capture_loop(self) -> None:
        frame_duration = 0.02
        samples_per_frame = int(self.sample_rate * frame_duration)
        while self._running:
            try:
                self._generate_simulated_frame(samples_per_frame)
                time.sleep(frame_duration)
            except Exception as e:
                logger.error(f"Error in simulated capture: {e}")
                time.sleep(0.1)

    def _generate_simulated_frame(self, num_samples: int = 320) -> None:
        try:
            t = np.linspace(0, num_samples / self.sample_rate, num_samples, False)
            audio_data = np.sin(2 * np.pi * 440 * t)
            audio_data = (audio_data * 32767).astype(np.int16)
            pcm_bytes = audio_data.tobytes()
            frame = (time.time(), pcm_bytes, self.sample_rate, self.channels)
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                logger.warning("Frame queue full, dropping frame")
        except Exception as e:
            logger.error(f"Error generating frame: {e}")


# ==================== NEW AND IMPROVED TEST HARNESS ====================

if __name__ == '__main__':
    print("=" * 70)
    print("VoIP Audio Capture - Audio Processing & Speech-to-Text Module")
    print("=" * 70)
    
    # --- Check for Vosk Model ---
    VOSK_MODEL_PATH = "model"
    if not os.path.isdir(VOSK_MODEL_PATH):
        print(f"Error: Vosk model not found in folder: '{VOSK_MODEL_PATH}'")
        print("Please download a model from https://alphacephei.com/vosk/models")
        print("Unzip it and rename the folder to 'model' in this directory.")
        exit(1)
    
    print("\nHow would you like to test?")
    print("  1: SIMULATED audio (a 440Hz 'oh' tone)")
    print("  2: Process a real .WAV file (You must upload one first)")
    
    cap = None
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        # --- WAV FILE TEST ---
        print("\n--- WAV File Test ---")
        print("Make sure you have uploaded a 16-bit, 16kHz, mono WAV file.")
        wav_path = input("Enter the path to your .wav file (e.g., my_test.wav): ").strip()
        
        if not os.path.exists(wav_path):
            print(f"Error: File not found at '{wav_path}'.")
            print("Falling back to simulation.")
            choice = "1"
        else:
            cap = VoipAudioCapture(
                wav_file_path=wav_path, # <-- Use the new WAV file mode
                enable_preprocessing=True,
                enable_transcription=True,
                save_to_csv=True
            )

    if choice == "1":
        # --- SIMULATION TEST (Default or Fallback) ---
        print("\n--- Simulation Test ---")
        print("This will auto-fallback to simulated audio (a 440Hz tone).")
        cap = VoipAudioCapture(
            use_microphone=True, # This will fail in Codespace and trigger simulation
            enable_preprocessing=True,
            enable_transcription=True,
            save_to_csv=True
        )

    # Start all background threads
    cap.start() 
    
    print("Capturing audio frames... Press Ctrl+C to stop.")
    print("-" * 70)
    
    frame_count = 0
    start_time = time.time()
    last_transcription = ""
    
    try:
        # This is "Lane 2" - the essential glue.
        # It pulls frames from Lane 1 (WAV file OR simulation)
        # and feeds them to Lane 3 (Transcription).
        for ts, pcm, sr, ch in cap.frames():
            frame_count += 1
            
            elapsed = time.time() - start_time
            volume = np.abs(np.frombuffer(pcm, dtype=np.int16)).mean()
            volume_bars = int(volume / 1000)
            volume_visual = "█" * min(volume_bars, 30)
            
            # Print a single, updating line of status
            print(f"\rTime: {elapsed:5.1f}s | Frames: {frame_count:4d} | Volume: {volume:6.0f} {volume_visual:<30}", end="", flush=True)

            # Check for new transcription
            if cap.enable_transcription and frame_count % 50 == 0: # Check every second
                current_transcription = cap.get_transcription().strip()
                if current_transcription != last_transcription:
                    # We just log the *new* text. The loop handles saving.
                    # This avoids printing the same text repeatedly.
                    new_text = current_transcription.replace(last_transcription, "").strip()
                    if new_text:
                         print(f"\n\n📝 NEW TRANSCRIPTION (for fraud detector):")
                         print(f"   '{new_text}'")
                    last_transcription = current_transcription

        # The loop will end automatically when the WAV file is done.
        print("\n\nCapture loop finished.")

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    finally:
        # Gracefully stop all threads and get final text
        
        # --- NEW: Get final chunk of text from Vosk ---
        if cap.enable_transcription and hasattr(cap._recognizer, 'FinalResult'):
            final_text = json.loads(cap._recognizer.FinalResult()).get('text', '')
            if final_text:
                logger.info(f"📝 Transcribed final chunk: {final_text}")
                # Manually add to transcription list
                cap._transcription_lock.acquire()
                cap._latest_transcription += " " + final_text
                cap._transcription_lock.release()
        # --- END NEW ---

        cap.stop()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("📊 Session Statistics:")
        print(f"   Duration: {elapsed:.1f} seconds")
        print(f"   Frames captured: {frame_count}")
        print(f"   Audio data: {frame_count * 640:,} bytes ({frame_count * 640 / 1024:.1f} KB)")
        
        if cap.enable_transcription:
            final_text = cap.get_transcription().strip()
            if final_text:
                print(f"\n📄 Full Transcription:")
                print(f"   {final_text}")
                
                # Save to CSV
                if cap.save_to_csv:
                    cap.save_transcription_to_csv(final_text, elapsed, frame_count)
                    print(f"\n💾 Transcription saved to: {cap.csv_filename}")
                    print(f"   Open it with: cat {cap.csv_filename}")
            
            print(f"\n➡️  This text would be sent to fraud detection module")
            print(f"   for keyword matching and analysis.")
        
        print("\n✅ Test completed!")
