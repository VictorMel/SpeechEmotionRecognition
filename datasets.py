"""Datasets module for Emotion Detection.

Provides:
- EmotionDataset: feature-on-demand with caching.
- StreamingEmotionDataset: windowed audio streaming for online inference.

Future Extensions:
- Multi-label / continuous emotion dimensions (valence, arousal).
- Balanced sampling / class-weight auto computation.
- Distillation support (teacher embeddings retrieval).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object  # Fallback for type hints

from audio_features import extract_with_cache, load_audio

@dataclass
class DatasetConfig:
    data_root: str = ""  # Must be provided explicitly
    feature_name: str = "mfcc"
    target_sr: int = 16000
    cache_dir: str = "cache_features/mfcc"
    feature_params: Dict = None
    file_ext: str = ".wav"
    label_map: Optional[Dict[str, int]] = None  # Optional external label mapping
    recursive: bool = True
    fixed_frames: Optional[int] = None  # Pad / truncate time dimension to this length if provided

class EmotionDataset(Dataset):
    """Dataset that loads audio files, extracts features (with caching), and returns tensor features + label.

    Directory layout expectation (simple form):
        root/
          angry/*.wav
          happy/*.wav
          neutral/*.wav
          ... class subfolders define labels.

    Extend:
      - Override `list_files` for CSV/meta-driven dataset.
      - Add custom label parsing logic.
    """
    def __init__(self, config: DatasetConfig, transform: Optional[Callable] = None):
        self.config = config
        self.transform = transform
        self.paths, self.labels, self.label2idx = self._discover()

    def debug_getmeta(self):
        return {"paths": self.paths, "labels": self.labels, "label2idxs": self.label2idx}

    def _discover(self) -> Tuple[List[str], List[int], Dict[str,int]]:
        label2idx: Dict[str,int] = {} if self.config.label_map is None else dict(self.config.label_map)
        paths: List[str] = []
        labels: List[int] = []
        root = self.config.data_root
        for dirpath, dirnames, filenames in os.walk(root):
            if not self.config.recursive and (dirpath != root):
                continue
            rel = os.path.relpath(dirpath, root)
            if rel == '.':  # root itself
                continue
            label = rel.split(os.sep)[0]
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            for f in filenames:
                if f.endswith(self.config.file_ext):
                    paths.append(os.path.join(dirpath, f))
                    labels.append(label2idx[label])
        return paths, labels, label2idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        feats = extract_with_cache(
            path,
            self.config.feature_name,
            self.config.cache_dir,
            target_sr=self.config.target_sr,
            params=self.config.feature_params or {},
        )
        # feats shape: (F, Tvar) or similar
        if self.config.fixed_frames is not None:
            Fdim, Tdim = feats.shape
            Ttarget = self.config.fixed_frames
            if Tdim < Ttarget:
                padded = np.zeros((Fdim, Ttarget), dtype=feats.dtype)
                padded[:, :Tdim] = feats
                feats = padded
            elif Tdim > Ttarget:
                feats = feats[:, :Ttarget]
        x = torch.tensor(feats, dtype=torch.float32)
        # Standard shape: (C, T) or (F, T); add channel dimension for CNN if needed
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(label, dtype=torch.long)

class StreamingEmotionDataset(Dataset):
    """Yields sequential windows of audio for streaming inference training/testing.
    Each item: (window_features, label, meta_dict)
    """
    def __init__(self, config: DatasetConfig, window_seconds: float = 2.0, hop_seconds: float = 0.5, max_windows: Optional[int] = None):
        self.config = config
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.max_windows = max_windows
        self.paths, self.labels, self.label2idx = EmotionDataset(config)._discover()

    def __len__(self):
        # Approximate upper bound if max_windows not set
        if self.max_windows is not None:
            return len(self.paths) * self.max_windows
        return len(self.paths) * int( (max(self.window_seconds, self.hop_seconds) + 1) )

    def __getitem__(self, idx: int):
        file_index = idx // (self.max_windows or 1)
        path = self.paths[file_index]
        label = self.labels[file_index]
        wav, sr = load_audio(path, target_sr=self.config.target_sr)
        win_len = int(self.window_seconds * sr)
        hop_len = int(self.hop_seconds * sr)
        start = (idx % (self.max_windows or 1)) * hop_len
        end = start + win_len
        if end > len(wav):
            end = len(wav)
        slice_wav = wav[start:end]
        feats = extract_with_cache(
            path,
            self.config.feature_name,
            self.config.cache_dir,
            target_sr=self.config.target_sr,
            params=self.config.feature_params or {},
        )
        # NOTE: For simplicity we ignore window alignment in features; future improvement: recompute on slice.
        x = torch.tensor(feats, dtype=torch.float32)
        meta = {"path": path, "start_sample": start, "end_sample": end, "sr": sr}
        return x, torch.tensor(label, dtype=torch.long), meta

if __name__ == "__main__":
    # Quick smoke test (will fail gracefully if torch not installed)
    try:
        cfg = DatasetConfig()
        ds = EmotionDataset(cfg)
        print(f"Discovered {len(ds)} files across {len(ds.label2idx)} labels: {ds.label2idx}")
    except Exception as e:
        print(f"Dataset initialization failed (expected if placeholder path): {e}")
