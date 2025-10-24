"""
Owner: Config Team
Responsibility: Model selection, paths and inference tuning defaults.

This file contains example settings for selecting and loading Hugging Face models for
English and Chinese scam detection. On Raspberry Pi 5 prefer quantized or small models.

Integration:
 - `detection.model_inference.InferenceEngine` reads these settings to load model(s)

"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Default HF model ids (replace with tuned checkpoints)
    TEXT_MODEL_EN: str = 'distilbert-base-uncased'  # small English model
    TEXT_MODEL_ZH: str = 'bert-base-chinese'        # example Chinese model

    # Path to locally cached models (prefer on-disk to avoid downloads on Pi)
    LOCAL_MODEL_DIR: str = '/opt/models'  # ensure this exists and is writable by app

    # Confidence threshold (0-100) above which alerts are raised
    ALERT_THRESHOLD_PERCENT: float = 75.0

    # Device string (e.g., 'cpu' or GPU device) - tune for AI Hat
    DEVICE: str = 'cpu'


if __name__ == '__main__':
    print('ModelConfig example', ModelConfig())
