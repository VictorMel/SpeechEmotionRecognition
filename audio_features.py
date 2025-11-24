"""Audio feature extraction module for Emotion Detection.

Design Goals:
- Centralize all audio loading & feature engineering in one place.
- Easy extension: add new feature builders by registering functions.
- Caching support: avoid recomputation for large datasets.
- Minimal external dependencies: prefer torchaudio > librosa when possible.

Future Extensions:
- GPU accelerated feature extraction (torch.compile / batched processing).
- Streaming windowed extraction for real-time inference.
- On-device (edge) reduced precision support (int8 / float16 path).
"""
from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
try:
    import torch
    import torchaudio
except ImportError:  # Allow feature enumeration without full deps
    torch = None
    torchaudio = None

# Removed legacy DATA_ROOT_PLACEHOLDER; rely on explicit --data-root or YAML config.

###############################################################################
# Registry Pattern
###############################################################################
FeatureFn = Callable[[np.ndarray, int, Dict], np.ndarray]
_FEATURE_REGISTRY: Dict[str, FeatureFn] = {}

def register_feature(name: str) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator to register new feature extraction function.
    Usage:
    @register_feature("chroma")
    def build_chroma(wav: np.ndarray, sr: int, params: Dict) -> np.ndarray: ...
    """
    def decorator(fn: FeatureFn) -> FeatureFn:
        if name in _FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' already registered")
        _FEATURE_REGISTRY[name] = fn
        return fn
    return decorator

###############################################################################
# Utility Helpers
###############################################################################

def _ensure_torch():
    if torchaudio is None:
        raise ImportError("torchaudio not installed. Please install before using this feature module.")


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _cache_path(cache_dir: str, key_parts: List[str]) -> str:
    key = "__".join(key_parts)
    return os.path.join(cache_dir, key + ".npy")


def list_features() -> List[str]:
    return sorted(_FEATURE_REGISTRY.keys())

###############################################################################
# Core Loading
###############################################################################

def load_audio(path: str, target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load audio file returning waveform and sampling rate.
    Abstracts backend differences (torchaudio vs fall-back librosa).
    """
    if torchaudio is not None:
        wav, sr = torchaudio.load(path)
        if mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = torchaudio.functional.resample(wav, sr, target_sr) if sr != target_sr else wav
        return wav.squeeze(0).numpy(), target_sr
    else:
        import librosa  # Local import to avoid hard dep when listing features
        wav, sr = librosa.load(path, sr=target_sr, mono=mono)
        return wav, sr

###############################################################################
# Feature Implementations
###############################################################################
@register_feature("mfcc")
def build_mfcc(wav: np.ndarray, sr: int, params: Dict) -> np.ndarray:
    """MFCC feature with optional delta/acceleration.
    params:
        n_mfcc: int
        add_delta: bool
        add_delta2: bool
        win_length: int (samples)
        hop_length: int (samples)
        log_energy: bool
    Returns: np.ndarray shape (feature_dim, time)
    """
    n_mfcc = params.get("n_mfcc", 40)
    win_length = params.get("win_length", int(0.025 * sr))
    hop_length = params.get("hop_length", int(0.010 * sr))
    add_delta = params.get("add_delta", True)
    add_delta2 = params.get("add_delta2", True)
    log_energy = params.get("log_energy", True)

    if torchaudio is not None:
        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1 << (win_length - 1).bit_length(), "hop_length": hop_length, "win_length": win_length},
        )(wav_t)
        feats = mfcc.squeeze(0).numpy()
    else:
        import librosa
        melspec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1 << (win_length - 1).bit_length(), hop_length=hop_length, win_length=win_length)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspec), sr=sr, n_mfcc=n_mfcc)
        feats = mfcc
    if log_energy:
        energy = np.log(np.maximum(1e-8, np.sum(feats**2, axis=0)))
        feats[0, :] = energy
    out = [feats]
    if add_delta:
        import librosa
        out.append(librosa.feature.delta(feats))
    if add_delta2:
        import librosa
        out.append(librosa.feature.delta(feats, order=2))
    return np.concatenate(out, axis=0)

@register_feature("mel")
def build_mel(wav: np.ndarray, sr: int, params: Dict) -> np.ndarray:
    """Mel Spectrogram (log scaled)."""
    n_fft = params.get("n_fft", 1024)
    hop_length = params.get("hop_length", 256)
    n_mels = params.get("n_mels", 128)
    if torchaudio is not None:
        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)(wav_t)
        mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
        return mel.squeeze(0).numpy()
    else:
        import librosa
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel)
        return mel_db

###############################################################################
# High-Level API
###############################################################################

def extract(feature_name: str, wav: np.ndarray, sr: int, params: Optional[Dict] = None) -> np.ndarray:
    params = params or {}
    if feature_name not in _FEATURE_REGISTRY:
        raise KeyError(f"Feature '{feature_name}' not registered. Available: {list_features()}")
    return _FEATURE_REGISTRY[feature_name](wav, sr, params)


def extract_from_path(path: str, feature_name: str, *, target_sr: int = 16000, params: Optional[Dict] = None) -> np.ndarray:
    wav, sr = load_audio(path, target_sr=target_sr)
    return extract(feature_name, wav, sr, params)


def extract_with_cache(path: str, feature_name: str, cache_dir: str, *, target_sr: int = 16000, params: Optional[Dict] = None, force: bool = False) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    params = params or {}
    key_parts = [feature_name, str(target_sr), os.path.basename(path)] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    cache_file = _cache_path(cache_dir, key_parts)
    if (not force) and os.path.isfile(cache_file):
        return np.load(cache_file)
    feats = extract_from_path(path, feature_name, target_sr=target_sr, params=params)
    np.save(cache_file, feats)
    return feats

###############################################################################
# CLI Helpers (Optional Future)
###############################################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build cached audio features for emotion dataset.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root (subfolders = labels)")
    parser.add_argument("--feature", type=str, default="mfcc", choices=list_features())
    parser.add_argument("--cache-dir", type=str, default="cache_features/mfcc")
    parser.add_argument("--pattern", type=str, default=".wav", help="Filename suffix to include")
    parser.add_argument("--target-sr", type=int, default=16000)
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise SystemExit(f"--data-root '{args.data_root}' does not exist.")

    paths = []
    for root, _, files in os.walk(args.data_root):
        for f in files:
            if f.endswith(args.pattern):
                paths.append(os.path.join(root, f))
    print(f"Found {len(paths)} files. Building {args.feature} cache -> {args.cache_dir}")
    params = {"n_mfcc": 40, "add_delta": True, "add_delta2": True}
    for i, p in enumerate(paths):
        feats = extract_with_cache(p, args.feature, args.cache_dir, target_sr=args.target_sr, params=params)
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(paths)}")
    print("Done.")
