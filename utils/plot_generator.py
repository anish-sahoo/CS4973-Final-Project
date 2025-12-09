import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.visualizer import TrainingVisualizer

def main():
    """Generate plot from metrics.json using TrainingVisualizer. Use this if 
    you ever modify the logic for visualizer after a training run and 
    want to use new plots."""
    
    metrics_path = 'runs/metrics.json'
    output_path = 'runs/generated_plot.png'

    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    visualizer = TrainingVisualizer(log_dir='runs', tb_enabled=False)
    visualizer.metrics = metrics

    visualizer.plot(save_path=output_path)

if __name__ == '__main__':
    main()