# Raspberry Pi Scam Detection System (scaffold)

This repository contains a scaffold for an AI-powered scam detection system that runs on Raspberry Pi 5 with an AI Hat.

Architecture split
- `frontend/` - Expo-managed React Native app (placeholder UI). Use Expo Go to run the mobile frontend.
- `backend/` - documentation for the Python backend (existing code currently lives at the repository root and `src/`).

Purpose
- Monitor VOIP phone calls and email (IMAP) for potential fraud targeting elderly nursing home residents.
- Provide real-time alerts (audio + visual) and a lightweight dashboard for caregivers.

What is included
- `frontend/` - minimal Expo app scaffold (placeholder UI).
- `src/` - Python package containing modules for audio, email, detection (training + inference), alerts, web dashboard, and configuration.
- `requirements.txt` - list of libraries to install (see notes below).
- `main.py` - example orchestrator wiring components for a simple demo.

How to start (development)
Backend (Python)
1. Create a Python virtual environment on the Pi and activate it.
2. Install dependencies (some packages require platform-specific wheels on the Pi).

  pip install -r requirements.txt

3. Edit configuration in `src/config/*_config.py` for your environment (IMAP creds, model paths, GPIO pins).
4. Start the orchestrator (development):

  python main.py

Frontend (Expo)
1. Install Expo CLI locally if you plan to develop the mobile app:

  npm install -g expo-cli

2. Open the frontend folder and install dependencies:

  cd frontend
  npm install

3. Start Expo and open in Expo Go or an emulator:

  npm start

Notes and next steps
- This scaffold contains detailed inline comments and TODOs for each module owner. It is intentionally minimal and meant to be extended.
- I did not move your existing Python files. If you want me to relocate the backend code into `backend/` and update imports, say so and I'll perform the relocation.
- For production use you should:
  - Replace placeholder inference with actual Hugging Face model loading and quantization for Pi performance.
  - Use a proper process supervisor (systemd) and restrict privileges for network/IMAP access.
  - Add unit tests, CI, and hardware-in-the-loop tests for GPIO/audio.

# raspberry-pi-voice-system