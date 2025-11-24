# Emotion Detection (Refactored Structure)

This directory contains a modularized Speech Emotion Recognition (SER) pipeline.
Existing legacy notebooks & educational NumPy code were moved to `old/`.

## Directory Layout
- `audio_features.py` – Central feature extraction (MFCC, Mel). Extend via `@register_feature`.
- `datasets.py` – Dataset + streaming dataset abstractions with caching.
- `models/` – Pluggable model definitions (`fc`, `resnet`, `lstm`). Factory in `models/__init__.py`.
- `training/loops.py` – Unified training loop (AMP, early stopping, distillation hook).
- `experiments/run_experiment.py` – CLI for config‑driven experiments.
- `configs/` – YAML experiment specs; edit dataset root before running.
- `old/` – Legacy files preserved (do **not** delete until migration stable).

## Quick Start
```powershell
# (Optional) Create feature cache first (speeds training)
python audio_features.py --data-root D:/DATA/EmotionDataset --feature mfcc --cache-dir cache_features/mfcc

# Train experiment from config
python experiments/run_experiment.py --config configs/resnet_mfcc.yaml --run-name resnet_mfcc_v1
```

## Adding a New Feature
```python
@register_feature("chroma")
def build_chroma(wav: np.ndarray, sr: int, params: Dict) -> np.ndarray:
    import librosa
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr)
    return chroma
```
Use in config: `feature_type: chroma`.

## Adding a New Model
Implement class in `models/your_model.py` and register in `models/__init__.py`:
```python
from .your_model import YourNet
_MODEL_REGISTRY["yournet"] = YourNet
```

## Distillation / Ensemble Hooks
- Distillation: pass `teacher_model` & `distill_weight` to `train_loop` (future CLI flag to be added).
- Ensemble: create wrapper that sums logits from multiple `build_model()` instances before loss.

## Edge Deployment Notes
- Ensure model forward uses only ONNX‑supported ops.
- Avoid dynamic control flow in new models.
- After training: `torch.onnx.export(model, example_input, 'emotion.onnx', opset_version=17)`.

## Legacy Preservation
- Do not edit files under `old/`; migrate logic gradually.
- If a notebook still imports `DeepLearningPyTorch`, keep original until replaced by `training/loops.py`.

## Next Steps (Optional Enhancements)
- Add validation/test split manifest for reproducibility.
- Implement class weighting or focal loss for imbalanced datasets.
- Add checkpoint resume CLI.
- Create ensemble runner script.

## Placeholder Paths
Replace `D:/DATA/EmotionDataset` in configs & `DATA_ROOT_PLACEHOLDER` when local dataset path is known.
