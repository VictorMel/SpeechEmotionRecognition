"""Model factory for Emotion Detection.

Supports plug-and-play creation via build_model(name, **kwargs).
Future:
- EnsembleModel wrapper combining multiple sub-model logits.
- Distillation hooks (teacher outputs capture).
"""
from typing import Any, Dict

from .fc_heads import SimpleFC
from .cnn_resnet import ResNetSpectrogram
from .lstm_models import LSTMClassifier

_MODEL_REGISTRY = {
    "fc": SimpleFC,
    "resnet": ResNetSpectrogram,
    "lstm": LSTMClassifier,
}

def available_models():
    return sorted(_MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs) -> Any:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {available_models()}")
    return _MODEL_REGISTRY[name](**kwargs)
