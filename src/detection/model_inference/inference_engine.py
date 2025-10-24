# Owner: Samya
# Responsibility: Real-time inference for both audio (via speech-to-text) and email text classification.
# Goals:
# - Load Hugging Face pre-trained models (Chinese/English) and perform classification returning confidence scores
# - Provide unified API that returns { 'scam_probability': float, 'explain': optional }
# Integration points:
# - Receives transcripts from audio pipeline (`audio_preprocessor` / STT service)
# - Receives parsed email text from `email.email_parser`
# - Sends alert events to `alerts.alert_logger` / `alerts.audio_alert_player`
# Testing requirements:
# - Unit tests with mocked HF model to verify output shape and confidence scaling to 0-100%

import typing as t
import numpy as np

# Dependencies: transformers, torch, sentencepiece (for some HF models)


class InferenceEngine:
    """
    High-level inference API.

    Constructor args:
        text_model_name: HF model id for text classification
        device: 'cpu' or 'cuda' or device string for AI Hat

    Public methods:
        - classify_text(text: str) -> dict: {'scam_probability': 0-100, 'raw_score': 0-1}
        - classify_transcript(transcript: str) -> dict (same schema)
        - classify_email(parsed_email) -> dict
        - load_model(path_or_name)

    TODOs:
        - Implement model loading with transformers.AutoModelForSequenceClassification
        - Add batching, caching, and warm-start logic
        - Provide language detection and route to appropriate model (Chinese vs English)
    """

    def __init__(self, text_model_name: str = 'bert-base-uncased', device: str = 'cpu'):
        self.text_model_name = text_model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name_or_path: str = None) -> None:
        """Load HF model and tokenizer. Use model_name_or_path or default.

        NOTE: On Raspberry Pi 5 with AI Hat, prefer quantized or small models for performance.
        """
        # TODO: call transformers.AutoTokenizer.from_pretrained and AutoModelForSequenceClassification
        self.model = 'loaded-model-placeholder'
        self.tokenizer = 'loaded-tokenizer-placeholder'

    def _score_to_percent(self, score: float) -> float:
        """Convert 0-1 score to 0-100 percentage and apply calibration if needed."""
        return float(np.clip(score * 100.0, 0.0, 100.0))

    def classify_text(self, text: str) -> dict:
        """Classify input text and return scam probability.

        Returns:
            {'scam_probability': float, 'raw_score': float}
        """
        # Placeholder naive heuristic until model is loaded
        raw_score = 0.01 if len(text) == 0 else min(0.5, len(text) / 1000.0)
        return {'scam_probability': self._score_to_percent(raw_score), 'raw_score': raw_score}

    def classify_transcript(self, transcript: str) -> dict:
        """Optionally perform language detection and classify the transcript."""
        return self.classify_text(transcript)

    def classify_email(self, parsed_email) -> dict:
        text = (parsed_email.subject or '') + '\n' + (parsed_email.text or '')
        return self.classify_text(text)


if __name__ == '__main__':
    e = InferenceEngine()
    print(e.classify_text('This is a suspicious request to transfer money'))
