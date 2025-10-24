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

# External dependencies (add to requirements.txt):
# - pjsua (PJSIP Python binding) or `twilio` for cloud-based calls
# - pyaudio for local microphone capture fallback


class VoipAudioCapture:
    """
    Connects to a VOIP endpoint and yields raw audio frames.

    Responsibilities:
    - Connect to a SIP stack (pjsua) or Twilio Media Streams and receive RTP audio.
    - Provide a generator / callback interface to obtain frames for downstream processing.

    Constructor parameters:
    - sip_config: dict - configuration for SIP/pjsua (host, port, credentials)
    - use_twilio: bool - whether to prefer Twilio integration

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

    def __init__(self, sip_config: t.Optional[dict] = None, use_twilio: bool = False):
        self.sip_config = sip_config or {}
        self.use_twilio = use_twilio
        self._running = False

    def start(self) -> None:
        """Start background capture.

        This should initialize the VOIP client and start receiving audio frames.
        Implementation note: Prefer non-blocking libraries and hand off frames through a queue.

        Returns: None
        """
        # TODO: Initialize pjsua or Twilio client and callback handlers
        self._running = True

    def stop(self) -> None:
        """Stop capture and clean up resources."""
        # TODO: Gracefully stop client and free sockets
        self._running = False

    def frames(self) -> t.Generator[t.Tuple[float, bytes, int, int], None, None]:
        """Yield audio frames as tuples: (timestamp, pcm_bytes, sample_rate, channels).

        This generator should block until frames are available. Consumers should be
        able to iterate over it in a dedicated thread or async task.
        """
        # TODO: Implement generator backed by an internal queue populated by callbacks
        while self._running:
            # placeholder sleep/yield to avoid busy-loop in skeleton
            import time
            time.sleep(0.5)
            yield (time.time(), b'', 16000, 1)


if __name__ == '__main__':
    # Basic manual test harness (non-functional until real VOIP integration is implemented)
    cap = VoipAudioCapture()
    cap.start()
    for i, (ts, pcm, sr, ch) in enumerate(cap.frames()):
        print('frame', i, 'ts', ts, 'len', len(pcm), 'sr', sr, 'ch', ch)
        if i > 2:
            break
    cap.stop()
