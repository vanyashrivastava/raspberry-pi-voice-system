# Raspberry Pi Scam Detection System (scaffold)

This repository contains a scaffold for an AI-powered scam detection system that runs on Raspberry Pi 5 with an AI Hat.

Purpose
- Monitor VOIP phone calls and email (IMAP) for potential fraud targeting elderly nursing home residents.
- Provide real-time alerts (audio + visual) and a lightweight dashboard for caregivers.

What is included
- `src/` - Python package containing modules for audio, email, detection (training + inference), alerts, web dashboard, and configuration.
- `requirements.txt` - list of libraries to install (see notes below).
- `main.py` - example orchestrator wiring components for a simple demo.

How to start (development)
1. Create a Python virtual environment on the Pi and activate it.
2. Install dependencies (some packages require platform-specific wheels on the Pi).

    pip install -r requirements.txt

3. Edit configuration in `src/config/*_config.py` for your environment (IMAP creds, model paths, GPIO pins).
4. Start the orchestrator (development):

    python main.py

Notes and next steps
- This scaffold contains detailed inline comments and TODOs for each module owner. It is intentionally minimal and meant to be extended.
- For production use you should:
  - Replace placeholder inference with actual Hugging Face model loading and quantization for Pi performance.
  - Use a proper process supervisor (systemd) and restrict privileges for network/IMAP access.
  - Add unit tests, CI, and hardware-in-the-loop tests for GPIO/audio.
# raspberry-pi-voice-system