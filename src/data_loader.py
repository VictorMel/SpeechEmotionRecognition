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

from audio_features import extract_from_path, load_audio

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
    dataset_name: str = ""  # Name of the dataset (e.g., RAVDESS, TESS, CREMA-D, SAVEE)

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
        self.paths, self.labels = self._discover()

    def _extract_label(self, audio_file_path: str, dataset_name: str) -> int:
        """Extract label based on dataset-specific rules."""
        emotion = -1
        if dataset_name == 'ravdess-emotional-speech-audio':
            emotion = int(audio_file_path[7:8]) - 1
        elif dataset_name == 'toronto-emotional-speech-set-tess':
            if '_neutral' in audio_file_path:
                emotion = 0
            elif '_happy' in audio_file_path:
                emotion = 2
            elif '_sad' in audio_file_path:
                emotion = 3
            elif '_angry' in audio_file_path:
                emotion = 4
            elif '_fear' in audio_file_path:
                emotion = 5
            elif '_disgust' in audio_file_path:
                emotion = 6
            elif '_ps' in audio_file_path:
                emotion = 7
        elif dataset_name == 'cremad':
            if '_NEU_' in audio_file_path:
                emotion = 0
            elif '_HAP_' in audio_file_path:
                emotion = 2
            elif '_SAD_' in audio_file_path:
                emotion = 3
            elif '_ANG_' in audio_file_path:
                emotion = 4
            elif '_FEA_' in audio_file_path:
                emotion = 5
            elif '_DIS_' in audio_file_path:
                emotion = 6
        elif dataset_name == 'savee-database':
            if 'n' in audio_file_path:
                emotion = 0
            elif 'h' in audio_file_path:
                emotion = 2
            elif 'sa' in audio_file_path:
                emotion = 3
            elif 'a' in audio_file_path:
                emotion = 4
            elif 'f' in audio_file_path:
                emotion = 5
            elif 'd' in audio_file_path:
                emotion = 6
            elif 'su' in audio_file_path:
                emotion = 7
        return emotion

    def _discover(self) -> Tuple[List[str], List[int], Dict[str, int]]:
        paths: List[str] = []
        labels: List[int] = []
        root = self.config.data_root

        for dataset_dir in os.listdir(root):
            dataset_path = os.path.join(root, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue

            dataset_name = dataset_dir  # Assign dataset name based on directory

            for dirpath, dirnames, filenames in os.walk(dataset_path):
                if not self.config.recursive and (dirpath != dataset_path):
                    continue
                for f in filenames:
                    if f.endswith(self.config.file_ext):
                        full_path = os.path.join(dirpath, f)
                        label = self._extract_label(f, dataset_name)
                        if label != -1:  # Skip files with invalid labels
                            paths.append(full_path)
                            labels.append(label)
        return paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        feats = extract_from_path(
            path,
            self.config.feature_name, 
            target_sr=self.config.target_sr, 
            params=self.config.feature_params or {}
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
