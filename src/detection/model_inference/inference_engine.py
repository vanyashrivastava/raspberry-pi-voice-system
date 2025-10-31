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



from __future__ import annotations

import time
import threading
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try optional heavy dependencies (transformers + torch). If unavailable, use fallback.
_HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# Module-level singleton and lock
_DEFAULT_ENGINE: Optional["InferenceEngine"] = None
_ENGINE_LOCK = threading.Lock()


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class InferenceEngine:
    """
    Lightweight inference engine.

    Args:
        text_model_name: HF model id or path (optional). If None, engine will use a
                         lightweight placeholder until load_model is called.
        device: 'cpu' or 'cuda' (if torch available).
        max_length: truncation length for preprocessing.
    """

    def __init__(self, text_model_name: Optional[str] = None, device: str = "cpu", max_length: int = 256):
        self.text_model_name = text_model_name
        self.device = device
        self.max_length = max_length

        # Hugging Face objects (if loaded)
        self.tokenizer = None
        self.model = None

        # status metadata
        self._status: Dict[str, Any] = {"state": "unloaded", "last_error": None, "model_version": None}
        self._lock = threading.Lock()

    # -------- model lifecycle --------
    def load_model(self, model_name_or_path: Optional[str] = None, warmup_examples: int = 2) -> None:
        """
        Load tokenizer and classification model. If HF libs are available, use them.
        Otherwise record the provided model_name_or_path as model_version and keep
        the deterministic fallback behavior.
        """
        with self._lock:
            self._status = {"state": "loading", "last_error": None, "model_version": None}
            try:
                if model_name_or_path:
                    self.text_model_name = model_name_or_path

                if _HAS_TRANSFORMERS and self.text_model_name:
                    logger.info("Loading HF tokenizer/model: %s", self.text_model_name)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.text_model_name)
                    try:
                        if torch and self.device and self.device != "cpu":
                            self.model.to(self.device)
                    except Exception:
                        logger.debug("Could not move model to requested device; continuing on CPU.")
                    self.model.eval()
                    version = getattr(self.model.config, "name_or_path", None) or str(self.text_model_name)
                    self._status["model_version"] = version
                else:
                    # fallback: no heavy deps - treat as a lightweight "model"
                    self.tokenizer = None
                    self.model = None
                    self._status["model_version"] = str(self.text_model_name or "fallback-v0")

                self._status["state"] = "loaded"

                # Warm-up: run a few lightweight inferences to allocate buffers (best-effort)
                for _ in range(max(1, warmup_examples)):
                    try:
                        self.run_inference("warmup", _skip_timer=True)
                    except Exception:
                        pass

            except Exception as exc:
                self._status = {"state": "error", "last_error": str(exc), "model_version": None}
                logger.exception("Failed to load model")
                raise

    # -------- preprocessing --------
    def preprocess_input(self, text_or_audio: Union[str, Dict[str, Any]], return_tokens: bool = False) -> Dict[str, Any]:
        """
        Apply deterministic preprocessing. If tokenizer is available, produce token tensors.
        Accepts:
          - text string
          - dict with 'transcript' key
        """
        if text_or_audio is None:
            raise ValueError("input is None")

        if isinstance(text_or_audio, dict) and "transcript" in text_or_audio:
            text = text_or_audio.get("transcript", "")
        else:
            text = str(text_or_audio)

        text = text.strip()

        if len(text) > self.max_length:
            text = text[: self.max_length]

        out: Dict[str, Any] = {"text": text, "length": len(text)}

        if _HAS_TRANSFORMERS and self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            try:
                if self.device and self.device != "cpu":
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
            except Exception:
                pass
            out["tokens"] = tokens

        return out

    # -------- internal model forward (HF or fallback) --------
    def _forward_model(self, preprocessed: Dict[str, Any]) -> float:
        """
        Returns a raw_score in [0,1] representing probability of 'scam' class.
        If an HF model is present, uses softmax to pick positive-class prob.
        Otherwise uses a deterministic hash-based fallback.
        """
        text = preprocessed.get("text", "")

        if _HAS_TRANSFORMERS and self.model is not None and preprocessed.get("tokens") is not None:
            try:
                tokens = preprocessed["tokens"]
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    if probs.shape[0] == 1:
                        raw = float(probs[0])
                    elif probs.shape[0] >= 2:
                        raw = float(probs[1])
                    else:
                        raw = float(probs.max())
                    return max(0.0, min(1.0, raw))
            except Exception as e:
                logger.exception("HF model forward failed; falling back: %s", e)

        h = hashlib.blake2b(text.encode("utf-8") if isinstance(text, str) else b"", digest_size=8).digest()
        num = int.from_bytes(h, "big")
        raw = (num % 10000) / 10000.0
        if len(text) == 0:
            raw = 0.0
        return float(raw)

    # -------- postprocessing --------
    @staticmethod
    def _postprocess_raw(raw_score: float) -> Dict[str, Any]:
        prob = max(0.0, min(100.0, raw_score * 100.0))
        if prob >= 75.0:
            conf = "high"
        elif prob >= 40.0:
            conf = "medium"
        else:
            conf = "low"
        return {"scam_probability": round(prob, 3), "confidence": conf, "raw_score": round(float(raw_score), 6)}

    # -------- public inference methods --------
    def run_inference(self, input_data: Union[str, Dict[str, Any]], _skip_timer: bool = False) -> Dict[str, Any]:
        start_ms = _now_ms()
        try:
            if input_data is None:
                raise ValueError("input_data is None")

            if self._status.get("state") != "loaded":
                raise RuntimeError("Model not loaded. Call load_model() first.")

            pre = self.preprocess_input(input_data)
            raw = self._forward_model(pre)
            post = self._postprocess_raw(raw)

            if not _skip_timer:
                post["processing_time_ms"] = round(_now_ms() - start_ms, 3)
            else:
                post["processing_time_ms"] = 0.0

            post["model_version"] = self._status.get("model_version")
            post["status"] = "ok"
            post["error_message"] = None
            post["metadata"] = {"device": self.device, "input_length": pre.get("length", 0)}
            return post

        except Exception as e:
            logger.exception("Inference error")
            return {
                "scam_probability": 0.0,
                "raw_score": 0.0,
                "confidence": "low",
                "model_version": self._status.get("model_version"),
                "processing_time_ms": round(_now_ms() - start_ms, 3),
                "status": "error",
                "error_message": str(e),
                "metadata": {"device": self.device},
            }

    def batch_inference(self, input_list: List[Union[str, Dict[str, Any]]], batch_size: int = 8) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if _HAS_TRANSFORMERS and self.tokenizer is not None:
            i = 0
            n = len(input_list)
            while i < n:
                chunk = input_list[i : i + batch_size]
                texts = []
                for it in chunk:
                    if isinstance(it, dict) and "transcript" in it:
                        texts.append(it.get("transcript", ""))
                    else:
                        texts.append(str(it))
                try:
                    tokens = self.tokenizer(
                        texts,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    if torch and self.device != "cpu":
                        tokens = {k: v.to(self.device) for k, v in tokens.items()}
                    with torch.no_grad():
                        outputs = self.model(**tokens)
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=-1).cpu().numpy()
                        for j in range(probs.shape[0]):
                            if probs.shape[1] >= 2:
                                raw = float(probs[j, 1])
                            else:
                                raw = float(probs[j].max())
                            post = self._postprocess_raw(raw)
                            post["processing_time_ms"] = 0.0
                            post["model_version"] = self._status.get("model_version")
                            post["status"] = "ok"
                            post["error_message"] = None
                            post["metadata"] = {"device": self.device, "batch_index": i + j}
                            results.append(post)
                except Exception as e:
                    logger.exception("Batched HF forward failed, falling back to per-item inference: %s", e)
                    for item in chunk:
                        results.append(self.run_inference(item))
                i += batch_size
            return results

        for item in input_list:
            results.append(self.run_inference(item))
        return results

    def get_model_status(self) -> Dict[str, Any]:
        return dict(self._status)

    # convenience
    def classify_text(self, text: str) -> Dict[str, Any]:
        return self.run_inference(text)

    def classify_transcript(self, transcript: str) -> Dict[str, Any]:
        return self.run_inference(transcript)

    def classify_email(self, parsed_email: Any) -> Dict[str, Any]:
        if isinstance(parsed_email, dict):
            subject = parsed_email.get("subject", "")
            body = parsed_email.get("text", parsed_email.get("body", ""))
        else:
            subject = getattr(parsed_email, "subject", "")
            body = getattr(parsed_email, "text", "")
        combined = (subject or "") + "\n" + (body or "")
        return self.run_inference(combined)


# -------- module-level convenience single-engine helpers --------
def get_default_engine() -> InferenceEngine:
    global _DEFAULT_ENGINE
    with _ENGINE_LOCK:
        if _DEFAULT_ENGINE is None:
            _DEFAULT_ENGINE = InferenceEngine()
        return _DEFAULT_ENGINE


def load_model(model_name_or_path: Optional[str] = None, warmup_examples: int = 2) -> None:
    eng = get_default_engine()
    eng.load_model(model_name_or_path, warmup_examples=warmup_examples)


def run_inference(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    eng = get_default_engine()
    if eng.get_model_status().get("state") != "loaded":
        try:
            eng.load_model(eng.text_model_name or "fallback-v0", warmup_examples=1)
        except Exception:
            pass
    return eng.run_inference(input_data)


def batch_inference(input_list: List[Union[str, Dict[str, Any]]], batch_size: int = 8) -> List[Dict[str, Any]]:
    eng = get_default_engine()
    if eng.get_model_status().get("state") != "loaded":
        try:
            eng.load_model(eng.text_model_name or "fallback-v0", warmup_examples=1)
        except Exception:
            pass
    return eng.batch_inference(input_list, batch_size=batch_size)


def get_model_status() -> Dict[str, Any]:
    eng = get_default_engine()
    return eng.get_model_status()


# quick CLI demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    e = InferenceEngine()
    print("Status before load:", e.get_model_status())
    e.load_model(None)  # fallback model
    print("Status after load:", e.get_model_status())
    examples = [
        "Congratulations! You've won $1000. Click here to claim.",
        "Hi team, please find the project update attached.",
        "",
        "请点击链接领取您的奖品",
    ]
    for ex in examples:
        print(ex, "->", e.run_inference(ex))