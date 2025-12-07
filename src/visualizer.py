"""
Visualizer for training metrics.
Tracks training loss, validation loss, and saves graphs at regular intervals.
Integrates with tensorboard for real-time monitoring.
"""

import os
import matplotlib.pyplot as plt
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TrainingVisualizer:
    """Tracks training metrics and generates visualization plots."""

    def __init__(self, log_dir='runs', tb_enabled=True):
        """
        Initialize the visualizer.
        
        Args:
            log_dir: Directory to save plots and logs
            tb_enabled: Whether to enable tensorboard logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        self.tb_enabled = tb_enabled
        if tb_enabled:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        self.metrics_file = self.log_dir / 'metrics.json'
        self.plot_file = self.log_dir / 'training_plot.png'
        
        # Load existing metrics if available
        self._load_metrics()

    def _load_metrics(self):
        """Load metrics from previous runs if available."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                print(f"Could not load existing metrics: {e}")
                self.metrics = {'train_loss': [], 'val_loss': [], 'epoch': []}

    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Could not save metrics: {e}")

    def log_train_loss(self, loss, epoch, step=None):
        """Log training loss."""
        if self.tb_enabled and self.writer:
            if step is not None:
                self.writer.add_scalar('Loss/train_step', loss, step)
            else:
                self.writer.add_scalar('Loss/train', loss, epoch)

    def log_val_loss(self, loss, epoch):
        """Log validation loss."""
        if self.tb_enabled and self.writer:
            self.writer.add_scalar('Loss/val', loss, epoch)

    def log_epoch(self, epoch, train_loss, val_loss=None):
        """Log metrics for an epoch."""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
        self._save_metrics()

    def plot(self, save_path=None):
        """
        Generate and save training plot.
        
        Args:
            save_path: Path to save the plot. If None, uses default plot_file.
        """
        if save_path is None:
            save_path = self.plot_file
        
        plt.figure(figsize=(10, 6))
        
        if self.metrics['epoch'] and self.metrics['train_loss']:
            plt.plot(self.metrics['epoch'], self.metrics['train_loss'], 
                    label='Training Loss', marker='o', linestyle='-', linewidth=2)
        
        if self.metrics['epoch'] and self.metrics['val_loss']:
            plt.plot(self.metrics['epoch'], self.metrics['val_loss'], 
                    label='Validation Loss', marker='s', linestyle='--', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        try:
            plt.savefig(str(save_path), dpi=100, format='png')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()

    def plot_at_interval(self, epoch, interval):
        """Plot every N epochs."""
        if (epoch + 1) % interval == 0:
            self.plot()

    def flush(self):
        """Flush tensorboard writer."""
        if self.tb_enabled and self.writer:
            self.writer.flush()

    def close(self):
        """Close tensorboard writer."""
        if self.tb_enabled and self.writer:
            self.writer.close()

    def get_summary(self):
        """Get training summary."""
        summary = {
            'total_epochs': len(self.metrics['epoch']),
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'best_train_loss': min(self.metrics['train_loss']) if self.metrics['train_loss'] else None,
            'best_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None,
        }
        return summary
