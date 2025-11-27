"""
Visualizer utility that writes to TensorBoard and also saves PNG plots of tracked scalars at intervals.

Usage:
    vis = Visualizer(log_dir='runs/default', plot_save_interval=100)
    vis.add_scalar('train/loss', loss_value, step)
    vis.save_plots('runs/default/plots/step_100.png')
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, log_dir: str = 'runs', plot_save_interval: int = 100):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(str(self.log_dir))
        self._scalars = defaultdict(list)  # tag -> list of (step, value)
        self.plot_save_interval = plot_save_interval
        self.plots_dir = self.log_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def add_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        self._scalars[tag].append((step, value))

    def flush(self):
        self.writer.flush()

    def save_plots(self, prefix: str = None):
        # prefix allows including step or other context like 'step_100'
        for tag, values in self._scalars.items():
            if not values:
                continue
            steps, v = zip(*values)
            plt.figure(figsize=(6, 4))
            plt.plot(steps, v, marker='o')
            plt.xlabel('step')
            plt.ylabel(tag)
            plt.title(tag)
            plt.grid(True)

            fname = f"{tag.replace('/', '_')}"
            if prefix:
                fname = f"{prefix}_{fname}"
            out = self.plots_dir / f"{fname}.png"
            plt.tight_layout()
            plt.savefig(out)
            plt.close()

    def maybe_save_plots(self, step: int):
        if self.plot_save_interval and step % self.plot_save_interval == 0:
            self.save_plots(prefix=f"step{step}")

    def save_predictions_grid(self, imgs, preds, step: int, prefix: str = 'preds'):
        """Save a small grid of image predictions by plotting points onto the images.

        imgs: Tensor (B,C,H,W) normalized as in the dataset transforms
        preds: Tensor (B,2) normalized coords
        """
        import torchvision.utils as vutils
        import torch
        import numpy as np
        from PIL import Image, ImageDraw

        # Bring to cpu and denormalize if needed
        imgs = imgs.detach().cpu()
        preds = preds.detach().cpu()

        # Convert first N images to PIL, overlay preds, and save as a grid
        n = min(8, imgs.size(0))
        grid_images = []
        for i in range(n):
            img = imgs[i]
            # img is normalized with mean=0.5 std=0.5; convert back
            img = (img * 0.5) + 0.5
            img = img.clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
            pil = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil)
            w, h = pil.size
            gx = int(preds[i, 0].item() * w)
            gy = int(preds[i, 1].item() * h)
            r = 3
            draw.ellipse([(gx - r, gy - r), (gx + r, gy + r)], fill=(255, 0, 0))
            grid_images.append(pil)

        # Create a single wide image
        widths, heights = zip(*(im.size for im in grid_images))
        total_w = sum(widths)
        max_h = max(heights)
        combined = Image.new('RGB', (total_w, max_h))
        x = 0
        for im in grid_images:
            combined.paste(im, (x, 0))
            x += im.size[0]

        out = self.plots_dir / f"{prefix}_step{step}.png"
        combined.save(out)
