Backend (Python) for Raspberry Pi Voice System

The repository currently contains the backend Python code at the repository root and under `src/`.

Key files / locations:
- `main.py` - top-level entrypoint (if present)
- `requirements.txt` - Python dependencies
- `src/` - application source code (audio, alerts, detection, email, etc.)

Quick start (Python backend):

1. Create a virtual environment and activate it:

   python -m venv venv
   source venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Run the test email scanner (mock mode):

   python src/email/email_test_simple.py

Notes
- I did NOT move or delete any existing files. If you want me to physically move the code into `backend/`, tell me and I'll perform the relocation (I will update imports as necessary).
- I can also add a small Flask API wrapper to expose endpoints for the frontend to call (e.g., `GET /emails/latest`).
