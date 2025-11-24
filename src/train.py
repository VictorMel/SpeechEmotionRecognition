"""CLI entry point to train emotion detection models using modular pipeline.

Example:
    python experiments/run_experiment.py --config configs/resnet_mfcc.yaml
"""
from __future__ import annotations
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from data_loader import EmotionDataset, DatasetConfig
from models import build_model, available_models
from loops import train_loop, TrainConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run-name', type=str, default='exp')
    parser.add_argument('--max-samples', type=int, default=0, help='Limit number of samples for quick debug run')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging (shapes, timings)')
    args = parser.parse_args()

    cfg_dict = load_config(args.config)

    # Dataset
    ds_cfg = DatasetConfig(
        data_root=cfg_dict['data']['root'],
        feature_name=cfg_dict['data']['feature_type'],
        target_sr=cfg_dict['data'].get('sr', 16000),
        cache_dir=cfg_dict['data'].get('cache_dir', f"cache_features/{cfg_dict['data']['feature_type']}"),
        feature_params=cfg_dict['data'].get('feature_params', {}),
        file_ext=cfg_dict['data'].get('ext', '.wav'),
        fixed_frames=cfg_dict['data'].get('fixed_frames', None),
    )
    if args.debug:
        print('[DEBUG] Initializing EmotionDataset ...')
    ds_full = EmotionDataset(ds_cfg)
    if args.max_samples > 0 and args.max_samples < len(ds_full):
        subset_indices = torch.arange(args.max_samples)
        ds_subset = torch.utils.data.Subset(ds_full, subset_indices)
        ds_work = ds_subset
        if args.debug:
            print(f"[DEBUG] Using subset of {args.max_samples} samples for quick run.")
    else:
        ds_work = ds_full
    print(f"Discovered {len(ds_work)} files")
    
    # Simple split
    indices = torch.randperm(len(ds_work))
    n = len(indices)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    ds_train = torch.utils.data.Subset(ds_work, train_idx)
    ds_val = torch.utils.data.Subset(ds_work, val_idx)
    ds_test = torch.utils.data.Subset(ds_work, test_idx)

    batch_size = cfg_dict['train']['batch_size']
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=cfg_dict['train'].get('num_workers', 0))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=cfg_dict['train'].get('num_workers', 0))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=cfg_dict['train'].get('num_workers', 0))

    # Model
    model_name = cfg_dict['model']['name']
    if model_name not in available_models():
        raise ValueError(f"Model {model_name} not available. Choices: {available_models()}")
    model = build_model(model_name, **cfg_dict['model'].get('params', {}))

    # Training config
    tcfg = TrainConfig(
        epochs=cfg_dict['train']['epochs'],
        lr=cfg_dict['train']['optimizer']['lr'],
        weight_decay=cfg_dict['train']['optimizer'].get('weight_decay', 0.0),
        amp=cfg_dict['train'].get('amp', True),
        accumulate_steps=cfg_dict['train'].get('accumulate_steps', 1),
        early_patience=cfg_dict['train'].get('early_patience', 8),
    )

    criterion = torch.nn.CrossEntropyLoss()

    tb_writer = SummaryWriter(log_dir=cfg_dict['logging']['log_dir']) if (SummaryWriter and cfg_dict['logging'].get('tensorboard', True)) else None

    print("Starting training loop...")
    if args.debug:
        # Peek one batch
        for bx, by in dl_train:
            print('[DEBUG] Batch shape:', tuple(bx.shape), 'Labels shape:', tuple(by.shape))
            break
    try:
        result = train_loop(
            model,
            dl_train,
            dl_val,
            tcfg,
            criterion,
            optimizer=None,
            metric_fn=lambda logits, y: (logits.argmax(dim=1) == y).float().mean(),
            run_name=args.run_name,
            checkpoint_dir=cfg_dict['logging']['checkpoint_dir'],
            tb_writer=tb_writer,
        )
        print("Training complete.", result)
    except Exception as e:
        print("[ERROR] Training aborted:", repr(e))

if __name__ == '__main__':
    main()
