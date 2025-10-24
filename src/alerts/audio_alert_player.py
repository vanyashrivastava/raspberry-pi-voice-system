# Owner: Siddhant
# Responsibility: Play audible alerts via USB speaker when a scam is suspected.
# Goals:
# - Provide configurable alert sounds and TTS fallback for dynamic messages
# - Ensure non-blocking playback and respect system volume
# Integration points:
# - Triggered by `detection.model_inference` when scam_probability passes threshold
# - Interacts with `visual_indicators` for combined multimodal alerts
# Testing requirements:
# - Integration tests with audio hardware (manual or CI with hardware stubs)

import typing as t

# Dependencies: pyaudio or sounddevice, pyttsx3 (for offline TTS fallback)


class AudioAlertPlayer:
    """
    Play alerts (beeps, pre-recorded messages, or TTS) in a non-blocking way.

    API:
        - play_beep(level='warning')
        - play_message_tts(text)
        - stop()

    TODOs:
        - Implement actual audio playback using sounddevice or pyaudio
        - Provide volume control and health checks for attached USB speaker
    """

    def __init__(self):
        pass

    def play_beep(self, level: str = 'warning') -> None:
        """Play a short beep. `level` can map to different tones/lengths."""
        # TODO: implement non-blocking playback
        print(f'play_beep({level})')

    def play_message_tts(self, text: str) -> None:
        """Speak a dynamic message using offline TTS (pyttsx3) as fallback."""
        # TODO: integrate pyttsx3 and ensure it's non-blocking
        print('TTS:', text)

    def stop(self) -> None:
        """Stop any ongoing playback."""
        # TODO
        pass


if __name__ == '__main__':
    p = AudioAlertPlayer()
    p.play_beep()
    p.play_message_tts('This is a test alert')
