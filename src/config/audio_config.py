"""
Owner: Config Team
Responsibility: Provide example audio configuration for microphone, sample rates, buffer sizes.

This file contains example settings for the audio subsystem. Owners (Oma) should adapt these
to their platform and hardware. Keep values conservative for Raspberry Pi 5.

Integration points:
 - `audio.voip_audio_capture` uses `SIP` and device selection settings
 - `audio.audio_preprocessor` uses `TARGET_SR` and frame sizes

Testing requirements:
 - Verify that `PYAUDIO_DEVICE_INDEX` matches `arecord -l` output on device

Example usage:
    from src.config.audio_config import AudioConfig
    cfg = AudioConfig()
    print(cfg.TARGET_SR)

"""

from dataclasses import dataclass


@dataclass
class AudioConfig:
    # Target sample rate for ML models (Hz). 16000 is a common choice for speech models.
    TARGET_SR: int = 16000

    # Number of channels to normalize to (1 = mono)
    CHANNELS: int = 1

    # Frame length expected from VOIP in milliseconds for buffering
    FRAME_MS: int = 20

    # pyaudio device index (None = default). Use `arecord -l` or `pyaudio` device list to find index.
    PYAUDIO_DEVICE_INDEX = None

    # VOIP-related defaults
    SIP_USER: str = ''
    SIP_PASS: str = ''
    SIP_SERVER: str = ''


if __name__ == '__main__':
    print('AudioConfig example:', AudioConfig())
