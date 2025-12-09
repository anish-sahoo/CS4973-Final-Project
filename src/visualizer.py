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
            'val_error': [],  # Angular error in degrees
            'val_acc_5': [],  # Accuracy within 5 degrees
            'val_acc_10': [],  # Accuracy within 10 degrees
            'epoch': [],
            'step_losses': [],  # Track loss at each step
            'steps': []  # Track step numbers
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
        if step is not None:
            self.metrics['step_losses'].append(loss)
            self.metrics['steps'].append(step)
            
        if self.tb_enabled and self.writer:
            if step is not None:
                self.writer.add_scalar('Loss/train_step', loss, step)
            else:
                self.writer.add_scalar('Loss/train', loss, epoch)

    def log_val_loss(self, loss, epoch):
        """Log validation loss."""
        if self.tb_enabled and self.writer:
            self.writer.add_scalar('Loss/val', loss, epoch)
    
    def log_eval_metrics(self, mean_error, acc_5, acc_10, epoch):
        """Log evaluation metrics."""
        self.metrics['val_error'].append(mean_error)
        self.metrics['val_acc_5'].append(acc_5)
        self.metrics['val_acc_10'].append(acc_10)
        
        if self.tb_enabled and self.writer:
            self.writer.add_scalar('Eval/Mean_Angular_Error', mean_error, epoch)
            self.writer.add_scalar('Eval/Accuracy_5deg', acc_5, epoch)
            self.writer.add_scalar('Eval/Accuracy_10deg', acc_10, epoch)

    def log_epoch(self, epoch, train_loss, val_loss=None):
        """Log metrics for an epoch."""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
        self._save_metrics()

    def plot(self, save_path=None):
        """
        Generate and save training plots.
        
        Args:
            save_path: Path to save the plot. If None, uses default plot_file.
        """
        if save_path is None:
            save_path = self.plot_file
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        
        # Plot 1: Loss vs Epoch
        if self.metrics['epoch'] and self.metrics['train_loss']:
            ax1.plot(self.metrics['epoch'], self.metrics['train_loss'], 
                    label='Training Loss', marker='o', linestyle='-', linewidth=2)
        
        if self.metrics['epoch'] and self.metrics['val_loss']:
            ax1.plot(self.metrics['epoch'], self.metrics['val_loss'], 
                    label='Validation Loss', marker='s', linestyle='--', linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Progress (Epoch-level)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss vs Training Steps
        if self.metrics['steps'] and self.metrics['step_losses']:
            ax2.plot(self.metrics['steps'], self.metrics['step_losses'], 
                    label='Training Loss', color='blue', alpha=0.3, linewidth=0.5)
            
            # Add a smoothed version with confidence intervals
            if len(self.metrics['step_losses']) > 100:
                window_size = min(100, len(self.metrics['step_losses']) // 10)
                smoothed, lower_bound, upper_bound = self._smooth_with_confidence(
                    self.metrics['step_losses'], window_size
                )
                steps_smoothed = self.metrics['steps'][:len(smoothed)]
                
                # Plot smoothed line
                ax2.plot(steps_smoothed, smoothed, 
                        label='Smoothed Loss', color='red', linewidth=2)
                
                # Plot 95% confidence interval
                ax2.fill_between(steps_smoothed, lower_bound, upper_bound, 
                                color='red', alpha=0.2, 
                                label='95% Confidence Interval')
        
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Progress (Step-level)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(top=0.2)  # Limit y-axis upper bound to 0.2
        
        # Plot 3: Evaluation Metrics
        if self.metrics['val_error']:
            eval_epochs = [i for i, e in enumerate(self.metrics['epoch']) if i < len(self.metrics['val_error'])]
            ax3_twin = ax3.twinx()
            
            # Plot angular error on left axis
            line1 = ax3.plot(eval_epochs, self.metrics['val_error'], 
                           label='Mean Angular Error', marker='o', color='red', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Angular Error (degrees)', fontsize=12, color='red')
            ax3.tick_params(axis='y', labelcolor='red')
            
            # Plot accuracies on right axis
            line2 = ax3_twin.plot(eval_epochs, self.metrics['val_acc_5'], 
                                 label='Acc @ 5°', marker='s', color='green', linewidth=2)
            line3 = ax3_twin.plot(eval_epochs, self.metrics['val_acc_10'], 
                                 label='Acc @ 10°', marker='^', color='blue', linewidth=2)
            ax3_twin.set_ylabel('Accuracy (%)', fontsize=12, color='blue')
            ax3_twin.tick_params(axis='y', labelcolor='blue')
            ax3_twin.set_ylim(80, 100)  # Limit accuracy axis from 80% to 100%
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, fontsize=10, loc='upper left')
            ax3.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            plt.savefig(str(save_path), dpi=100, format='png')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()
    
    def _smooth(self, values, window_size):
        """Simple moving average smoothing."""
        import numpy as np
        smoothed = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            smoothed.append(np.mean(window))
        return smoothed
    
    def _smooth_with_confidence(self, values, window_size, confidence=0.95):
        """Moving average smoothing with confidence intervals.
        
        Args:
            values: List of values to smooth
            window_size: Size of the moving window
            confidence: Confidence level (default 0.95 for 95% CI)
        
        Returns:
            smoothed, lower_bound, upper_bound
        """
        import numpy as np
        from scipy import stats
        
        smoothed = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            mean = np.mean(window)
            std = np.std(window)
            n = len(window)
            
            # Calculate 95% confidence interval using t-distribution
            # t-value for 95% confidence with n-1 degrees of freedom
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_value * (std / np.sqrt(n))
            
            smoothed.append(mean)
            lower_bounds.append(mean - margin)
            upper_bounds.append(mean + margin)
        
        return smoothed, lower_bounds, upper_bounds

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
