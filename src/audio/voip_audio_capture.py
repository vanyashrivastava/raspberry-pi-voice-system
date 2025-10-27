# Owner: Oma
# Responsibility: Implement VOIP-side audio capture integration and bridging to local audio pipelines.
# Goals:
# - Provide a stable stream of raw audio frames (PCM float32 or int16) from VOIP calls.
# - Support hooking into pjsua (PJSIP) or Twilio media streams when available on the Pi.
# Integration points:
# - Exposes audio frames consumed by `audio.audio_preprocessor.AudioPreprocessor`.
# - Emits events or pushes to `audio.audio_stream_handler.AudioStreamHandler` for buffering.
# Testing requirements:
# - Unit tests should mock the VOIP stack and validate that frames are emitted with correct sample rate, channels.

import typing as t
import queue
import threading
import time
import numpy as np
import logging

# External dependencies (add to requirements.txt):
# - pjsua (PJSIP Python binding) or `twilio` for cloud-based calls
# - pyaudio for local microphone capture fallback
# - numpy for audio processing

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoipAudioCapture:
    """
    Connects to a VOIP endpoint and yields raw audio frames.
    
    Responsibilities:
    - Connect to a SIP stack (pjsua) or Twilio Media Streams and receive RTP audio.
    - Provide a generator / callback interface to obtain frames for downstream processing.
    
    Constructor parameters:
    - sip_config: dict - configuration for SIP/pjsua (host, port, credentials)
    - use_twilio: bool - whether to prefer Twilio integration
    - use_microphone: bool - capture from computer's microphone (for testing)
    - buffer_size: int - size of the internal audio frame queue
    
    Public methods:
    - start(): Start background capture (non-blocking)
    - stop(): Stop capture and release resources
    - frames(): Generator yielding (timestamp, pcm_bytes, sample_rate, channels)
    
    TODOs for implementation:
    - Implement pjsua client integration with media callbacks.
    - Implement Twilio Media Streams client (WebSocket) to receive audio.
    - Normalize output to a consistent format (e.g., 16kHz, mono, 16-bit PCM).
    - Add reconnection/backoff logic and error handling for flaky networks.
    
    Example usage:
        cap = VoipAudioCapture(sip_config={'user':'1000', 'pass':'secret'})
        cap.start()
        for ts, pcm, sr, ch in cap.frames():
            # feed into preprocessor
            pass
    """
    
    def __init__(
        self, 
        sip_config: t.Optional[dict] = None, 
        use_twilio: bool = False,
        use_microphone: bool = False,
        buffer_size: int = 100
    ):
        """
        Initialize the VoIP audio capture system.
        
        Args:
            sip_config: Dictionary with SIP credentials {'user': '...', 'pass': '...', 'host': '...'}
            use_twilio: If True, use Twilio instead of PJSIP
            use_microphone: If True, capture from computer's microphone (great for testing!)
            buffer_size: Maximum number of audio frames to buffer in memory
        """
        self.sip_config = sip_config or {}
        self.use_twilio = use_twilio
        self.use_microphone = use_microphone
        self.buffer_size = buffer_size
        
        # Internal state
        self._running = False
        self._capture_thread = None
        
        # Queue to store audio frames between capture and consumption
        # This allows the VOIP callback to run independently from your processing
        self._frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Audio format settings (standardized for consistency)
        self.sample_rate = 16000  # 16kHz is standard for voice
        self.channels = 1  # Mono audio
        self.sample_width = 2  # 16-bit audio (2 bytes per sample)
        self.chunk_size = 320  # 20ms of audio at 16kHz
        
        # VOIP client objects (will be initialized in start())
        self._pjsua_lib = None
        self._pjsua_account = None
        self._twilio_client = None
        self._pyaudio_instance = None
        self._pyaudio_stream = None
        
        mode = "microphone" if use_microphone else ("twilio" if use_twilio else "pjsua")
        logger.info(f"VoipAudioCapture initialized (mode={mode})")
    
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
        
        # Start the appropriate capture method based on configuration
        if self.use_microphone:
            self._start_microphone_capture()
        elif self.use_twilio:
            self._start_twilio_capture()
        else:
            self._start_pjsua_capture()
        
        logger.info("VoIP audio capture started")
    
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
        
        # Clear any remaining frames in the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("VoIP audio capture stopped")
    
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
                # This prevents the generator from blocking forever when stopping
                frame_data = self._frame_queue.get(timeout=1.0)
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
            # Callback function will be called whenever new audio is available
            self._pyaudio_stream = self._pyaudio_instance.open(
                format=pyaudio.paInt16,  # 16-bit audio
                channels=self.channels,   # Mono
                rate=self.sample_rate,    # 16kHz
                input=True,               # This is an input stream (microphone)
                frames_per_buffer=self.chunk_size,  # 20ms chunks
                stream_callback=self._microphone_callback  # Called automatically with audio
            )
            
            # Start the stream
            self._pyaudio_stream.start_stream()
            logger.info(f"🎤 Microphone capture started! Speak into your mic...")
            logger.info(f"   Sample rate: {self.sample_rate} Hz")
            logger.info(f"   Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f}ms)")
            
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            self._start_simulated_capture()  # Fallback to simulation
    
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
        """
        Callback function called by PyAudio when new audio is available.
        
        This runs in a separate thread managed by PyAudio, so we just
        need to put the data into our queue for processing.
        
        Args:
            in_data: Raw audio bytes from microphone
            frame_count: Number of frames captured
            time_info: Timing information
            status: Status flags (for detecting errors)
        
        Returns:
            Tuple of (None, pyaudio.paContinue) to keep stream running
        """
        import pyaudio
        
        # Check for errors
        if status:
            logger.warning(f"PyAudio status flags: {status}")
        
        # Create frame tuple with timestamp and audio data
        frame = (
            time.time(),      # Current timestamp
            in_data,          # Raw PCM bytes from mic
            self.sample_rate, # 16000 Hz
            self.channels     # 1 channel
        )
        
        # Add to queue (non-blocking)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            logger.warning("Frame queue full, dropping microphone frame")
        
        # Return paContinue to keep the stream running
        return (None, pyaudio.paContinue)
    
    # ==================== PJSUA Implementation ====================
    
    def _start_pjsua_capture(self) -> None:
        """
        Initialize and start PJSUA (PJSIP) client.
        
        PJSIP is an open-source SIP stack that can handle VoIP calls.
        This method sets up the library and registers a callback to
        receive audio frames during calls.
        """
        try:
            import pjsua as pj
        except ImportError:
            logger.error("pjsua not installed. Install with: pip install pjsua")
            # Fall back to simulation mode for development
            self._start_simulated_capture()
            return
        
        try:
            # Create PJSUA library instance
            self._pjsua_lib = pj.Lib()
            
            # Initialize library with default settings
            self._pjsua_lib.init(
                ua_cfg=pj.UAConfig(),  # User Agent config
                log_cfg=pj.LogConfig(level=3, callback=self._pjsua_log_callback),
                media_cfg=pj.MediaConfig()  # Media config
            )
            
            # Start the library (required before making calls)
            self._pjsua_lib.start()
            
            # Create a transport (UDP on default port 5060)
            transport = self._pjsua_lib.create_transport(pj.TransportType.UDP)
            logger.info(f"PJSUA listening on {transport.info().host}:{transport.info().port}")
            
            # Create and register SIP account if credentials provided
            if self.sip_config.get('user') and self.sip_config.get('host'):
                acc_cfg = pj.AccountConfig(
                    domain=self.sip_config['host'],
                    username=self.sip_config['user'],
                    password=self.sip_config.get('pass', '')
                )
                self._pjsua_account = self._pjsua_lib.create_account(acc_cfg)
                logger.info(f"Registered SIP account: {self.sip_config['user']}@{self.sip_config['host']}")
            
            # Start background thread to handle PJSUA events and capture audio
            self._capture_thread = threading.Thread(target=self._pjsua_capture_loop, daemon=True)
            self._capture_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start PJSUA: {e}")
            self._start_simulated_capture()  # Fallback
    
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
        """
        Background thread that polls PJSUA for events and captures audio.
        
        PJSUA requires regular polling to process network events.
        This loop also simulates audio capture from active calls.
        """
        import pjsua as pj
        
        while self._running:
            try:
                # Poll for PJSUA events (handles SIP messages, RTP packets, etc.)
                self._pjsua_lib.handle_events(50)  # 50ms timeout
                
                # Check for active calls and capture audio
                # In a real implementation, you'd hook into PJSUA's media callbacks
                # For now, we simulate audio capture when there are active calls
                calls = self._pjsua_lib.enum_calls()
                if calls:
                    # Generate simulated audio frame
                    # TODO: Replace with actual audio capture from call media stream
                    self._generate_simulated_frame()
                
            except Exception as e:
                logger.error(f"Error in PJSUA capture loop: {e}")
                time.sleep(0.1)
    
    def _pjsua_log_callback(self, level: int, message: str, length: int) -> None:
        """Callback for PJSUA log messages."""
        if level <= 3:  # Only log important messages
            logger.debug(f"PJSUA: {message.strip()}")
    
    # ==================== Twilio Implementation ====================
    
    def _start_twilio_capture(self) -> None:
        """
        Initialize and start Twilio Media Streams client.
        
        Twilio Media Streams provides real-time access to call audio
        via WebSocket connection. This is useful for cloud-based deployments.
        """
        try:
            from twilio.twiml.voice_response import VoiceResponse, Start
            import websocket
        except ImportError:
            logger.error("Twilio not installed. Install with: pip install twilio websocket-client")
            self._start_simulated_capture()
            return
        
        # TODO: Implement Twilio WebSocket connection
        # For now, fall back to simulation
        logger.warning("Twilio integration not fully implemented, using simulation")
        self._start_simulated_capture()
    
    def _stop_twilio_capture(self) -> None:
        """Clean up Twilio resources."""
        if self._twilio_client:
            # TODO: Close WebSocket connection
            self._twilio_client = None
            logger.info("Twilio stopped")
    
    # ==================== Simulation Mode (for testing) ====================
    
    def _start_simulated_capture(self) -> None:
        """
        Start simulated audio capture for testing without real VOIP.
        
        This generates synthetic audio frames at the correct rate,
        useful for development and testing without a real phone system.
        """
        logger.info("Starting simulated audio capture (for testing)")
        self._capture_thread = threading.Thread(target=self._simulated_capture_loop, daemon=True)
        self._capture_thread.start()
    
    def _simulated_capture_loop(self) -> None:
        """
        Background thread that generates simulated audio frames.
        
        Generates frames at ~20ms intervals (standard for VoIP)
        with a simple sine wave pattern for testing.
        """
        frame_duration = 0.02  # 20ms frames (standard for VoIP)
        samples_per_frame = int(self.sample_rate * frame_duration)
        
        while self._running:
            try:
                self._generate_simulated_frame(samples_per_frame)
                time.sleep(frame_duration)  # Simulate real-time capture
            except Exception as e:
                logger.error(f"Error in simulated capture: {e}")
                time.sleep(0.1)
    
    def _generate_simulated_frame(self, num_samples: int = 320) -> None:
        """
        Generate a simulated audio frame and add it to the queue.
        
        Args:
            num_samples: Number of samples to generate (default 320 = 20ms at 16kHz)
        """
        try:
            # Generate a sine wave (440 Hz = musical note A4)
            t = np.linspace(0, num_samples / self.sample_rate, num_samples, False)
            audio_data = np.sin(2 * np.pi * 440 * t)
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            pcm_bytes = audio_data.tobytes()
            
            # Create frame tuple
            frame = (
                time.time(),  # timestamp
                pcm_bytes,    # audio data
                self.sample_rate,  # 16000 Hz
                self.channels  # 1 channel (mono)
            )
            
            # Add to queue (non-blocking, drops frame if queue is full)
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                logger.warning("Frame queue full, dropping frame")
        
        except Exception as e:
            logger.error(f"Error generating frame: {e}")


# ==================== Main Test Harness ====================

if __name__ == '__main__':
    print("=== VoIP Audio Capture Test ===\n")
    print("Choose test mode:")
    print("1. Microphone capture (RECOMMENDED - speak into your mic!)")
    print("2. Simulated audio (generates 440Hz tone)")
    print("3. PJSUA (requires pjsua library)")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        print("\n🎤 Starting MICROPHONE capture...")
        print("Speak into your microphone! Press Ctrl+C to stop.\n")
        cap = VoipAudioCapture(use_microphone=True)
    elif choice == "2":
        print("\n🔊 Starting SIMULATED audio (440Hz tone)...\n")
        cap = VoipAudioCapture()
    else:
        print("\n📞 Starting PJSUA (will fallback to simulation if not installed)...\n")
        cap = VoipAudioCapture(sip_config={'user': '1000', 'pass': 'secret'})
    
    cap.start()
    
    print("Capturing audio frames... (showing first 10 frames)")
    print("-" * 70)
    
    try:
        frame_count = 0
        total_bytes = 0
        start_time = time.time()
        
        for ts, pcm, sr, ch in cap.frames():
            frame_count += 1
            total_bytes += len(pcm)
            
            # Calculate audio level (volume) for visualization
            audio_array = np.frombuffer(pcm, dtype=np.int16)
            volume = np.abs(audio_array).mean()
            volume_bars = int(volume / 1000)  # Scale for display
            volume_visual = "█" * min(volume_bars, 50)
            
            if frame_count <= 10:
                print(f"Frame {frame_count:3d}: {len(pcm):4d} bytes @ {sr:5d}Hz | "
                      f"Volume: {volume:6.0f} {volume_visual}")
            elif frame_count == 11:
                print("... continuing (volume meter only) ...")
            else:
                # Just show volume meter after first 10 frames
                print(f"\rFrame {frame_count:3d} | Volume: {volume:6.0f} {volume_visual}" + " " * 20, end="")
            
            # Optional: Stop after 100 frames (2 seconds) for quick test
            # if frame_count >= 100:
            #     break
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    finally:
        cap.stop()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"📊 Statistics:")
        print(f"   Frames captured: {frame_count}")
        print(f"   Total audio data: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
        print(f"   Duration: {elapsed:.1f} seconds")
        print(f"   Average rate: {frame_count/elapsed:.1f} frames/sec")
        print("\n✅ Test completed!")