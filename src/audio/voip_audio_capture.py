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
from datetime import datetime

# External dependencies (add to requirements.txt):
# - pjsua (PJSIP Python binding) or `twilio` for cloud-based calls
# - pyaudio for local microphone capture fallback
# - numpy for audio processing
# - scipy for audio preprocessing (filtering, resampling)
# - vosk for offline speech-to-text OR SpeechRecognition for online

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocesses raw audio frames for speech-to-text and feature extraction.
    
    Responsibilities:
    - Normalize volume levels
    - Remove background noise
    - Apply bandpass filter for speech frequencies
    - Resample if needed
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the audio preprocessor.
        
        Args:
            target_sample_rate: Target sample rate for output audio (default 16000 Hz)
        """
        self.target_sample_rate = target_sample_rate
        logger.info(f"AudioPreprocessor initialized (target_sr={target_sample_rate})")
    
    def process(self, audio_bytes: bytes, original_sample_rate: int) -> np.ndarray:
        """
        Process raw audio bytes and return cleaned audio array.
        
        Args:
            audio_bytes: Raw PCM audio bytes
            original_sample_rate: Original sampling rate of the audio
            
        Returns:
            Cleaned audio as numpy array (float32, normalized to -1.0 to 1.0)
        """
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 and normalize to -1.0 to 1.0 range
        audio = audio.astype(np.float32) / 32768.0
        
        # Resample if needed
        if original_sample_rate != self.target_sample_rate:
            audio = self._resample(audio, original_sample_rate)
        
        # Apply noise reduction
        audio = self._reduce_noise(audio)
        
        # Apply bandpass filter to keep only speech frequencies (300-3400 Hz)
        audio = self._bandpass_filter(audio)
        
        # Normalize volume
        audio = self._normalize_volume(audio)
        
        return audio
    
    def _resample(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            from scipy import signal
            num_samples = int(len(audio) * self.target_sample_rate / original_sr)
            return signal.resample(audio, num_samples)
        except ImportError:
            logger.warning("scipy not installed, skipping resample")
            return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Simple noise gate to remove low-level background noise."""
        threshold = 0.01  # Noise threshold
        audio[np.abs(audio) < threshold] = 0
        return audio
    
    def _bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to keep speech frequencies (300-3400 Hz)."""
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
        """Normalize audio volume to consistent level."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8  # Normalize to 80% of max
        return audio


class VoipAudioCapture:
    """
    Connects to a VOIP endpoint, yields raw audio frames, preprocesses audio, and transcribes speech.
    
    Responsibilities:
    - Connect to a SIP stack (pjsua) or Twilio Media Streams and receive RTP audio.
    - Provide a generator / callback interface to obtain frames for downstream processing.
    - Preprocess audio for optimal quality
    - Transcribe speech to text
    - Provide clean audio and text to downstream fraud detection modules
    
    Constructor parameters:
    - sip_config: dict - configuration for SIP/pjsua (host, port, credentials)
    - use_twilio: bool - whether to prefer Twilio integration
    - use_microphone: bool - capture from computer's microphone (for testing)
    - enable_preprocessing: bool - enable audio preprocessing (noise reduction, filtering)
    - enable_transcription: bool - enable speech-to-text
    - buffer_size: int - size of the internal audio frame queue
    
    Public methods:
    - start(): Start background capture (non-blocking)
    - stop(): Stop capture and release resources
    - frames(): Generator yielding (timestamp, pcm_bytes, sample_rate, channels)
    - get_transcription(): Get latest transcribed text
    - get_preprocessed_audio(): Get latest preprocessed audio chunk
    
    TODOs for implementation:
    - Implement pjsua client integration with media callbacks.
    - Implement Twilio Media Streams client (WebSocket) to receive audio.
    - Normalize output to a consistent format (e.g., 16kHz, mono, 16-bit PCM).
    - Add reconnection/backoff logic and error handling for flaky networks.
    
    Example usage:
        # Basic audio capture
        cap = VoipAudioCapture(use_microphone=True)
        cap.start()
        for ts, pcm, sr, ch in cap.frames():
            # Use raw audio
            pass
        
        # With preprocessing and transcription
        cap = VoipAudioCapture(
            use_microphone=True, 
            enable_preprocessing=True,
            enable_transcription=True
        )
        cap.start()
        
        # Get transcribed text for fraud detection module
        text = cap.get_transcription()
        
        # fraud_detector.check_for_scam(text)  # Someone else's code
    """
    
    def __init__(
        self, 
        sip_config: t.Optional[dict] = None, 
        use_twilio: bool = False,
        use_microphone: bool = False,
        enable_preprocessing: bool = True,
        enable_transcription: bool = False,
        save_to_csv: bool = True,
        csv_filename: str = "audio_transcripts.csv",
        buffer_size: int = 100
    ):
        """
        Initialize the VoIP audio capture system.
        
        Args:
            sip_config: Dictionary with SIP credentials {'user': '...', 'pass': '...', 'host': '...'}
            use_twilio: If True, use Twilio instead of PJSIP
            use_microphone: If True, capture from computer's microphone (great for testing!)
            enable_preprocessing: If True, preprocess audio (noise reduction, filtering)
            enable_transcription: If True, enable speech-to-text transcription
            save_to_csv: If True, save transcriptions to CSV file
            csv_filename: Name of CSV file to save transcriptions
            buffer_size: Maximum number of audio frames to buffer in memory
        """
        self.sip_config = sip_config or {}
        self.use_twilio = use_twilio
        self.use_microphone = use_microphone
        self.enable_preprocessing = enable_preprocessing
        self.enable_transcription = enable_transcription
        self.save_to_csv = save_to_csv
        self.csv_filename = csv_filename
        self.buffer_size = buffer_size
        
        # Internal state
        self._running = False
        self._capture_thread = None
        self._transcription_thread = None
        
        # Queue to store audio frames between capture and consumption
        self._frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Queue for speech recognition (larger chunks needed)
        self._transcription_queue = queue.Queue(maxsize=10)
        
        # Audio format settings (standardized for consistency)
        self.sample_rate = 16000  # 16kHz is standard for voice
        self.channels = 1  # Mono audio
        self.sample_width = 2  # 16-bit audio (2 bytes per sample)
        self.chunk_size = 320  # 20ms of audio at 16kHz
        
        # Preprocessing
        self._preprocessor = AudioPreprocessor(target_sample_rate=self.sample_rate) if enable_preprocessing else None
        self._latest_preprocessed_audio = None
        self._preprocessed_lock = threading.Lock()
        
        # Speech recognition
        self._recognizer = None
        self._latest_transcription = ""
        self._transcription_lock = threading.Lock()
        self._audio_buffer_for_transcription = []
        
        # VOIP client objects (will be initialized in start())
        self._pjsua_lib = None
        self._pjsua_account = None
        self._twilio_client = None
        self._pyaudio_instance = None
        self._pyaudio_stream = None
        
        # Initialize speech recognition if enabled
        if self.enable_transcription:
            self._init_speech_recognition()
        
        # Initialize CSV file if saving enabled
        if self.save_to_csv and self.enable_transcription:
            self._init_csv_file()
        
        mode = "microphone" if use_microphone else ("twilio" if use_twilio else "pjsua")
        logger.info(f"VoipAudioCapture initialized (mode={mode}, preprocessing={enable_preprocessing}, transcription={enable_transcription}, csv={save_to_csv})")
    
    def _init_csv_file(self) -> None:
        """Initialize CSV file for saving transcriptions."""
        try:
            # Check if file exists
            file_exists = os.path.isfile(self.csv_filename)
            
            # Create file with headers if it doesn't exist
            if not file_exists:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp',
                        'session_id',
                        'transcription',
                        'duration_seconds',
                        'audio_frames',
                        'mode'
                    ])
                logger.info(f"✅ Created CSV file: {self.csv_filename}")
            else:
                logger.info(f"✅ Using existing CSV file: {self.csv_filename}")
        
        except Exception as e:
            logger.error(f"Failed to initialize CSV file: {e}")
            self.save_to_csv = False
    
    def save_transcription_to_csv(self, transcription: str, duration: float, frame_count: int) -> None:
        """
        Save transcription to CSV file.
        
        Args:
            transcription: The transcribed text
            duration: Duration of the recording in seconds
            frame_count: Number of audio frames captured
        """
        if not self.save_to_csv:
            return
        
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    int(time.time()),  # Session ID (Unix timestamp)
                    transcription,
                    f"{duration:.2f}",
                    frame_count,
                    "microphone" if self.use_microphone else "voip"
                ])
            logger.info(f"💾 Saved transcription to {self.csv_filename}")
        
        except Exception as e:
            logger.error(f"Failed to save to CSV: {e}")
    
    def _init_speech_recognition(self) -> None:
        """Initialize speech recognition engine."""
        try:
            # Try Vosk first (offline, more accurate)
            from vosk import Model, KaldiRecognizer
            import os
            
            # Download model if needed: https://alphacephei.com/vosk/models
            model_path = "model"  # Path to Vosk model
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
        """Initialize Google Speech Recognition (online, requires internet)."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            logger.info("✅ Using Google Speech Recognition (online)")
        except ImportError:
            logger.error("❌ Neither Vosk nor SpeechRecognition installed!")
            logger.error("Install with: pip install vosk OR pip install SpeechRecognition")
            self.enable_transcription = False
    
    def get_transcription(self) -> str:
        """
        Get the latest transcribed text.
        
        This text should be passed to the fraud detection module for keyword analysis.
        
        Returns:
            Latest transcription as string
        """
        with self._transcription_lock:
            return self._latest_transcription
    
    def get_preprocessed_audio(self) -> t.Optional[np.ndarray]:
        """
        Get the latest preprocessed audio chunk.
        
        Returns:
            Preprocessed audio as numpy array, or None if preprocessing disabled
        """
        if not self.enable_preprocessing:
            return None
        
        with self._preprocessed_lock:
            return self._latest_preprocessed_audio
    
    def start(self) -> None:
        """
        Start background capture.
        
        This initializes the VOIP client and starts receiving audio frames
        in a background thread. Frames are placed into an internal queue
        for consumption via the frames() generator.
        
        Returns: None
        """
        if self._running:
            logger.warning("Capture already running")
            return
        
        self._running = True
        
        # Start transcription thread if enabled
        if self.enable_transcription:
            self._transcription_thread = threading.Thread(
                target=self._transcription_loop, 
                daemon=True
            )
            self._transcription_thread.start()
        
        # Start the appropriate capture method based on configuration
        if self.use_microphone:
            self._start_microphone_capture()
        elif self.use_twilio:
            self._start_twilio_capture()
        else:
            self._start_pjsua_capture()
        
        logger.info("✅ VoIP audio capture started")
    
    def stop(self) -> None:
        """
        Stop capture and clean up resources.
        
        This gracefully shuts down the VOIP client, stops background threads,
        and clears the frame queue.
        """
        if not self._running:
            return
        
        logger.info("Stopping VoIP audio capture...")
        self._running = False
        
        # Stop the appropriate client
        if self.use_microphone:
            self._stop_microphone_capture()
        elif self.use_twilio:
            self._stop_twilio_capture()
        else:
            self._stop_pjsua_capture()
        
        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        # Wait for transcription thread
        if self._transcription_thread and self._transcription_thread.is_alive():
            self._transcription_thread.join(timeout=2.0)
        
        # Clear any remaining frames in the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("✅ VoIP audio capture stopped")
    
    def frames(self) -> t.Generator[t.Tuple[float, bytes, int, int], None, None]:
        """
        Yield audio frames as tuples: (timestamp, pcm_bytes, sample_rate, channels).
        
        This generator blocks until frames are available. It should be consumed
        in a dedicated thread or async task. Each frame contains:
        - timestamp: Unix timestamp when frame was captured
        - pcm_bytes: Raw PCM audio data as bytes
        - sample_rate: Sampling rate in Hz (e.g., 16000)
        - channels: Number of audio channels (1=mono, 2=stereo)
        
        Yields:
            Tuple of (timestamp, pcm_bytes, sample_rate, channels)
        """
        while self._running:
            try:
                # Wait up to 1 second for a frame
                frame_data = self._frame_queue.get(timeout=1.0)
                
                # Preprocess audio if enabled
                if self.enable_preprocessing:
                    self._preprocess_audio(frame_data[1], frame_data[2])  # pcm_bytes, sample_rate
                
                # Buffer audio for transcription if enabled
                if self.enable_transcription:
                    self._buffer_audio_for_transcription(frame_data[1])  # pcm_bytes
                
                yield frame_data
            except queue.Empty:
                # No frames available, continue waiting
                continue
        
        # Yield any remaining frames after stopping
        while not self._frame_queue.empty():
            try:
                frame_data = self._frame_queue.get_nowait()
                yield frame_data
            except queue.Empty:
                break
    
    def _preprocess_audio(self, pcm_bytes: bytes, sample_rate: int) -> None:
        """
        Preprocess audio and store result.
        
        Args:
            pcm_bytes: Raw PCM audio bytes
            sample_rate: Sample rate of the audio
        """
        try:
            cleaned_audio = self._preprocessor.process(pcm_bytes, sample_rate)
            with self._preprocessed_lock:
                self._latest_preprocessed_audio = cleaned_audio
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
    
    def _buffer_audio_for_transcription(self, pcm_bytes: bytes) -> None:
        """
        Buffer audio for transcription. Accumulates ~1 second of audio before transcribing.
        
        Args:
            pcm_bytes: Raw PCM audio data
        """
        self._audio_buffer_for_transcription.append(pcm_bytes)
        
        # Transcribe every 1 second of audio (50 frames at 20ms each)
        if len(self._audio_buffer_for_transcription) >= 50:
            # Combine all buffered audio
            combined_audio = b''.join(self._audio_buffer_for_transcription)
            
            # Send to transcription queue
            try:
                self._transcription_queue.put_nowait(combined_audio)
            except queue.Full:
                pass  # Skip if queue is full
            
            # Clear buffer
            self._audio_buffer_for_transcription = []
    
    def _transcription_loop(self) -> None:
        """Background thread that transcribes audio to text."""
        logger.info("📝 Transcription thread started")
        
        while self._running:
            try:
                # Get audio chunk to transcribe (1 second chunks)
                audio_chunk = self._transcription_queue.get(timeout=1.0)
                
                # Transcribe based on recognizer type
                if hasattr(self._recognizer, 'AcceptWaveform'):
                    # Vosk recognizer
                    text = self._transcribe_vosk(audio_chunk)
                else:
                    # Google Speech Recognition
                    text = self._transcribe_google(audio_chunk)
                
                if text:
                    with self._transcription_lock:
                        self._latest_transcription += " " + text
                        # Keep only last 1000 characters
                        self._latest_transcription = self._latest_transcription[-1000:]
                    
                    logger.info(f"📝 Transcribed: {text}")
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")
    
    def _transcribe_vosk(self, audio_bytes: bytes) -> str:
        """Transcribe using Vosk (offline)."""
        try:
            if self._recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self._recognizer.Result())
                return result.get('text', '')
            else:
                partial = json.loads(self._recognizer.PartialResult())
                return partial.get('partial', '')
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            return ""
    
    def _transcribe_google(self, audio_bytes: bytes) -> str:
        """Transcribe using Google Speech Recognition (online)."""
        try:
            import speech_recognition as sr
            
            # Convert bytes to AudioData
            audio_data = sr.AudioData(audio_bytes, self.sample_rate, self.sample_width)
            
            # Transcribe
            text = self._recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return ""  # Speech not understood
        except sr.RequestError as e:
            logger.error(f"Google Speech API error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Google transcription error: {e}")
            return ""
    
    # ==================== MICROPHONE CAPTURE (for testing) ====================
    
    def _start_microphone_capture(self) -> None:
        """
        Start capturing audio from the computer's microphone using PyAudio.
        
        This is perfect for testing your fraud detection pipeline without
        needing a real phone system. Just speak into your mic!
        """
        try:
            import pyaudio
        except ImportError:
            logger.error("PyAudio not installed. Install with: pip install pyaudio")
            logger.info("Falling back to simulation mode...")
            self._start_simulated_capture()
            return
        
        try:
            # Initialize PyAudio
            self._pyaudio_instance = pyaudio.PyAudio()
            
            # List available input devices (helpful for debugging)
            logger.info("Available audio input devices:")
            for i in range(self._pyaudio_instance.get_device_count()):
                dev_info = self._pyaudio_instance.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:
                    logger.info(f"  [{i}] {dev_info['name']} - {dev_info['maxInputChannels']} channels")
            
            # Open audio stream from default microphone
            self._pyaudio_stream = self._pyaudio_instance.open(
                format=pyaudio.paInt16,  # 16-bit audio
                channels=self.channels,   # Mono
                rate=self.sample_rate,    # 16kHz
                input=True,               # This is an input stream (microphone)
                frames_per_buffer=self.chunk_size,  # 20ms chunks
                stream_callback=self._microphone_callback
            )
            
            # Start the stream
            self._pyaudio_stream.start_stream()
            logger.info(f"🎤 Microphone capture started!")
            logger.info(f"   Sample rate: {self.sample_rate} Hz")
            logger.info(f"   Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f}ms)")
            
            if self.enable_preprocessing:
                logger.info("   🔧 Audio preprocessing ENABLED")
            
            if self.enable_transcription:
                logger.info("   📝 Speech-to-text ENABLED")
            
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            self._start_simulated_capture()
    
    def _stop_microphone_capture(self) -> None:
        """Clean up PyAudio resources."""
        try:
            if self._pyaudio_stream:
                self._pyaudio_stream.stop_stream()
                self._pyaudio_stream.close()
                self._pyaudio_stream = None
            
            if self._pyaudio_instance:
                self._pyaudio_instance.terminate()
                self._pyaudio_instance = None
            
            logger.info("Microphone capture stopped")
        except Exception as e:
            logger.error(f"Error stopping microphone: {e}")
    
    def _microphone_callback(self, in_data, frame_count, time_info, status):
        """Callback function called by PyAudio when new audio is available."""
        import pyaudio
        
        if status:
            logger.warning(f"PyAudio status flags: {status}")
        
        # Create frame tuple
        frame = (
            time.time(),
            in_data,
            self.sample_rate,
            self.channels
        )
        
        # Add to queue
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            logger.warning("Frame queue full, dropping microphone frame")
        
        return (None, pyaudio.paContinue)
    
    # ==================== PJSUA Implementation ====================
    
    def _start_pjsua_capture(self) -> None:
        """Initialize and start PJSUA (PJSIP) client."""
        try:
            import pjsua as pj
        except ImportError:
            logger.error("pjsua not installed. Install with: pip install pjsua")
            self._start_simulated_capture()
            return
        
        try:
            self._pjsua_lib = pj.Lib()
            self._pjsua_lib.init(
                ua_cfg=pj.UAConfig(),
                log_cfg=pj.LogConfig(level=3, callback=self._pjsua_log_callback),
                media_cfg=pj.MediaConfig()
            )
            self._pjsua_lib.start()
            
            transport = self._pjsua_lib.create_transport(pj.TransportType.UDP)
            logger.info(f"PJSUA listening on {transport.info().host}:{transport.info().port}")
            
            if self.sip_config.get('user') and self.sip_config.get('host'):
                acc_cfg = pj.AccountConfig(
                    domain=self.sip_config['host'],
                    username=self.sip_config['user'],
                    password=self.sip_config.get('pass', '')
                )
                self._pjsua_account = self._pjsua_lib.create_account(acc_cfg)
                logger.info(f"Registered SIP account: {self.sip_config['user']}@{self.sip_config['host']}")
            
            self._capture_thread = threading.Thread(target=self._pjsua_capture_loop, daemon=True)
            self._capture_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start PJSUA: {e}")
            self._start_simulated_capture()
    
    def _stop_pjsua_capture(self) -> None:
        """Clean up PJSUA resources."""
        try:
            if self._pjsua_lib:
                self._pjsua_lib.destroy()
                self._pjsua_lib = None
            logger.info("PJSUA stopped")
        except Exception as e:
            logger.error(f"Error stopping PJSUA: {e}")
    
    def _pjsua_capture_loop(self) -> None:
        """Background thread that polls PJSUA for events."""
        import pjsua as pj
        
        while self._running:
            try:
                self._pjsua_lib.handle_events(50)
                calls = self._pjsua_lib.enum_calls()
                if calls:
                    self._generate_simulated_frame()
            except Exception as e:
                logger.error(f"Error in PJSUA capture loop: {e}")
                time.sleep(0.1)
    
    def _pjsua_log_callback(self, level: int, message: str, length: int) -> None:
        """Callback for PJSUA log messages."""
        if level <= 3:
            logger.debug(f"PJSUA: {message.strip()}")
    
    # ==================== Twilio Implementation ====================
    
    def _start_twilio_capture(self) -> None:
        """Initialize and start Twilio Media Streams client."""
        try:
            from twilio.twiml.voice_response import VoiceResponse, Start
            import websocket
        except ImportError:
            logger.error("Twilio not installed. Install with: pip install twilio websocket-client")
            self._start_simulated_capture()
            return
        
        logger.warning("Twilio integration not fully implemented, using simulation")
        self._start_simulated_capture()
    
    def _stop_twilio_capture(self) -> None:
        """Clean up Twilio resources."""
        if self._twilio_client:
            self._twilio_client = None
            logger.info("Twilio stopped")
    
    # ==================== Simulation Mode ====================
    
    def _start_simulated_capture(self) -> None:
        """Start simulated audio capture for testing."""
        logger.info("Starting simulated audio capture (for testing)")
        self._capture_thread = threading.Thread(target=self._simulated_capture_loop, daemon=True)
        self._capture_thread.start()
    
    def _simulated_capture_loop(self) -> None:
        """Background thread that generates simulated audio frames."""
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
        """Generate a simulated audio frame."""
        try:
            t = np.linspace(0, num_samples / self.sample_rate, num_samples, False)
            audio_data = np.sin(2 * np.pi * 440 * t)
            audio_data = (audio_data * 32767).astype(np.int16)
            pcm_bytes = audio_data.tobytes()
            
            frame = (
                time.time(),
                pcm_bytes,
                self.sample_rate,
                self.channels
            )
            
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                logger.warning("Frame queue full, dropping frame")
        
        except Exception as e:
            logger.error(f"Error generating frame: {e}")


# ==================== Main Test Harness ====================

if __name__ == '__main__':
    print("=" * 70)
    print("VoIP Audio Capture - Audio Processing & Speech-to-Text Module")
    print("=" * 70)
    print("\nThis module provides:")
    print("  ✅ Raw audio capture from microphone/phone")
    print("  ✅ Audio preprocessing (noise reduction, filtering)")
    print("  ✅ Speech-to-text transcription")
    print("  ➡️  Text output for fraud detection module (someone else's code)")
    print()
    print("Choose test mode:")
    print("1. Full pipeline (capture + preprocessing + transcription)")
    print("2. Capture + preprocessing only")
    print("3. Capture only (raw audio)")
    print("4. Simulated audio (no microphone)")
    
    choice = input("\nEnter choice (1/2/3/4) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        print("\n🎤 FULL PIPELINE: Capture + Preprocess + Transcribe")
        print("Speak into your microphone!")
        print("The transcribed text will be shown (for fraud detection module to analyze)\n")
        cap = VoipAudioCapture(
            use_microphone=True,
            enable_preprocessing=True,
            enable_transcription=True
        )
    elif choice == "2":
        print("\n🎤 CAPTURE + PREPROCESSING")
        print("Audio will be captured and cleaned (no transcription)\n")
        cap = VoipAudioCapture(
            use_microphone=True,
            enable_preprocessing=True,
            enable_transcription=False
        )
    elif choice == "3":
        print("\n🎤 RAW CAPTURE ONLY")
        print("Audio captured without any processing\n")
        cap = VoipAudioCapture(
            use_microphone=True,
            enable_preprocessing=False,
            enable_transcription=False
        )
    else:
        print("\n🔊 SIMULATED AUDIO (440Hz tone)\n")
        cap = VoipAudioCapture(
            enable_preprocessing=False,
            enable_transcription=False
        )
    
    cap.start()
    
    print("Capturing audio frames... Press Ctrl+C to stop.")
    print("-" * 70)
    
    try:
        frame_count = 0
        start_time = time.time()
        last_transcription = ""
        
        for ts, pcm, sr, ch in cap.frames():
            frame_count += 1
            
            # Calculate volume
            audio_array = np.frombuffer(pcm, dtype=np.int16)
            volume = np.abs(audio_array).mean()
            volume_bars = int(volume / 1000)
            volume_visual = "█" * min(volume_bars, 30)
            
            # Show status every 10 frames (200ms)
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"\rTime: {elapsed:5.1f}s | Frames: {frame_count:4d} | Volume: {volume:6.0f} {volume_visual:<30}", end="", flush=True)
            
            # Get transcription if enabled
            if cap.enable_transcription and frame_count % 50 == 0:  # Check every second
                current_transcription = cap.get_transcription().strip()
                
                # Only print if transcription changed
                if current_transcription and current_transcription != last_transcription:
                    print(f"\n\n📝 TRANSCRIPTION (for fraud detector):")
                    print(f"   '{current_transcription}'")
                    print()
                    last_transcription = current_transcription
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    finally:
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
        print("\nIntegration example:")
        print("  text = cap.get_transcription()")
        print("  fraud_detector.check_for_scam(text)  # Someone else's code")