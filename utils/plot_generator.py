import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def load_metrics(metrics_path):
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def smooth_with_confidence(values, window_size, confidence=0.95):
    """Moving average smoothing with confidence intervals.
    
    Applies smoothing and calculates confidence intervals based on 
    residuals from the smoothed line, not raw variance.
    """
    smoothed = []
    
    half_window = window_size // 2
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        window = values[start:end]
        smoothed.append(np.mean(window))
    
    residuals = [values[i] - smoothed[i] for i in range(len(values))]
    
    lower_bounds = []
    upper_bounds = []
    
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        window_residuals = residuals[start:end]
        
        std = np.std(window_residuals)
        n = len(window_residuals)
        
        if n > 1:
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_value * (std / np.sqrt(n))
        else:
            margin = 0
        
        lower_bounds.append(smoothed[i] - margin)
        upper_bounds.append(smoothed[i] + margin)

    return smoothed, lower_bounds, upper_bounds

def generate_plot(metrics, output_path):
    """Generate and save training plots."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

    # Plot 1: Loss vs Epoch
    if metrics.get('epoch') and metrics.get('train_loss'):
        ax1.plot(metrics['epoch'], metrics['train_loss'],
                label='Training Loss', marker='o', linestyle='-', linewidth=2)

    if metrics.get('epoch') and metrics.get('val_loss'):
        ax1.plot(metrics['epoch'], metrics['val_loss'],
                label='Validation Loss', marker='s', linestyle='--', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress (Epoch-level)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss vs Training Steps
    if metrics.get('steps') and metrics.get('step_losses'):
        ax2.plot(metrics['steps'], metrics['step_losses'],
                label='Training Loss', color='blue', alpha=0.3, linewidth=0.5)

        if len(metrics['step_losses']) > 100:
            window_size = min(100, len(metrics['step_losses']) // 10)
            smoothed, lower_bound, upper_bound = smooth_with_confidence(
                metrics['step_losses'], window_size
            )
            ax2.plot(metrics['steps'], smoothed,
                    label='Smoothed Loss', color='red', linewidth=2)
            ax2.fill_between(metrics['steps'], lower_bound, upper_bound,
                            color='red', alpha=0.2,
                            label='95% Confidence Interval')

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Progress (Step-level)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.1)

    # Plot 3: Evaluation Metrics
    if metrics.get('val_error'):
        eval_epochs = list(range(len(metrics['val_error'])))
        ax3_twin = ax3.twinx()

        # Plot angular error on left axis
        line1 = ax3.plot(eval_epochs, metrics['val_error'],
                       label='Mean Angular Error', marker='o', color='red', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Angular Error (degrees)', fontsize=12, color='red')
        ax3.tick_params(axis='y', labelcolor='red')
        ax3.set_ylim(0, 4)

        # Plot accuracies on right axis
        if metrics.get('val_acc_5'):
            line2 = ax3_twin.plot(eval_epochs, metrics['val_acc_5'],
                                 label='Acc @ 5°', marker='s', color='green', linewidth=2)
        if metrics.get('val_acc_10'):
            line3 = ax3_twin.plot(eval_epochs, metrics['val_acc_10'],
                                 label='Acc @ 10°', marker='^', color='blue', linewidth=2)
        ax3_twin.set_ylabel('Accuracy (%)', fontsize=12, color='blue')
        ax3_twin.tick_params(axis='y', labelcolor='blue')
        ax3_twin.set_ylim(87.5, 100)

        lines = line1 + (line2 if 'line2' in locals() else []) + (line3 if 'line3' in locals() else [])
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, fontsize=10, loc='lower right')
        ax3.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=100, format='png')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

def main():
    metrics_path = 'runs/metrics.json'
    output_path = 'runs/generated_plot.png'

    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    metrics = load_metrics(metrics_path)

    generate_plot(metrics, output_path)

if __name__ == '__main__':
    main()