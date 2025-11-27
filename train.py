"""
Main training script for the gaze tracker starter.

It supports:
- dataset CSV input or synthetic fallback
- device selection via `config.py`
- TensorBoard logging + PNG plot saving at intervals
- minimal training loop with checkpointing
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

from models.eye_gaze_net import build_model
from data.dataset import SimpleEyeDataset, RandomEyeDataset
from utils.visualizer import Visualizer
from utils.device import get_device
import config


def save_checkpoint(model, optimizer, epoch, prefix='checkpoint', out_dir='checkpoints'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{prefix}_epoch{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Saved checkpoint to {path}")


def train():
    device = get_device()
    print(f"Using device: {device}")
    # Datasets
    csv_path = getattr(config, 'CSV_PATH', None)
    if csv_path is not None and Path(csv_path).exists():
        train_ds = SimpleEyeDataset(csv_path)
    else:
        print('[train] CSV not provided or not found; using RandomEyeDataset fallback')
        n = 256 if not getattr(config, 'DEBUG', False) else 32
        train_ds = RandomEyeDataset(n=n)

    loader = DataLoader(train_ds, batch_size=getattr(config, 'BATCH_SIZE', 32), shuffle=True)

    # Model
    model = build_model()
    model.to(device)

    # Training niceties
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=getattr(config, 'LR', 1e-3))

    # Visualizer/TensorBoard
    vis = Visualizer(log_dir=getattr(config, 'LOG_DIR', 'runs'), plot_save_interval=getattr(config, 'PLOT_SAVE_INTERVAL', 100))

    # Training loop
    step = 0
    for epoch in range(1, getattr(config, 'EPOCHS', 10) + 1):
        model.train()
        running_loss = 0.0
        for i, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % getattr(config, 'LOG_INTERVAL', 10) == 0:
                avg_loss = running_loss / (i + 1)
                print(f"Epoch {epoch} Step {step} AvgLoss {avg_loss:.6f}")
                vis.add_scalar('train/loss', avg_loss, step)
                vis.flush()

            # Save plots periodically
            vis.maybe_save_plots(step)

            # Save prediction overlays periodically (first batch)
            if step % getattr(config, 'PLOT_SAVE_INTERVAL', 100) == 0:
                try:
                    vis.save_predictions_grid(imgs, preds, step)
                except Exception as e:
                    print(f"[train] Could not save prediction overlay: {e}")

            step += 1

        # Post-epoch checkpoint
        save_checkpoint(model, optimizer, epoch, prefix='gaze_net', out_dir=getattr(config, 'CHECKPOINT_DIR', 'checkpoints'))

    print('Training completed')


if __name__ == '__main__':
    # All runtime configuration comes from `config.py`. Edit `config.py` to change options.
    train()
