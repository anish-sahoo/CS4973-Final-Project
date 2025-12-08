import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random

import config
from datasets.mpiigaze_dataset import MPIIGazeDataset
from models.gaze_model import GazeNet
from utils import compute_angular_error
from infer_realtime import find_best_model

def evaluate():
    sample_size = config.EVAL_SAMPLE_SIZE
    batch_size = config.EVAL_BATCH_SIZE
    
    device = torch.device(config.DEVICE if config.DEVICE != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {config.CSV_PATH}...")
    dataset = MPIIGazeDataset(config.CSV_PATH)
    total_len = len(dataset)
    
    if sample_size > total_len:
        print(f"Requested sample size {sample_size} is larger than dataset size {total_len}. Using full dataset.")
        indices = list(range(total_len))
    else:
        print(f"Sampling {sample_size} random images from {total_len} total images.")
        indices = random.sample(range(total_len), sample_size)
    
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model_path = find_best_model()
    if not model_path:
        print("No model checkpoint found. Please train the model first.")
        return

    print(f"Loading model from {model_path}...")
    model = GazeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_errors = []

    print("Starting evaluation...")
    with torch.no_grad():
        for left_imgs, right_imgs, head, gaze in tqdm(loader):
            left_imgs = left_imgs.to(device)
            right_imgs = right_imgs.to(device)
            head = head.to(device)
            
            # Forward pass
            pred = model(left_imgs, right_imgs, head)
            
            # Move to CPU for metric calculation
            pred_np = pred.cpu().numpy()
            gaze_np = gaze.numpy()
            
            # Compute angular error
            errors = compute_angular_error(pred_np, gaze_np)
            all_errors.extend(errors)

    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    # Calculate accuracy at different thresholds
    all_errors = np.array(all_errors)
    acc_5 = np.mean(all_errors < 5.0) * 100
    acc_10 = np.mean(all_errors < 10.0) * 100
    acc_15 = np.mean(all_errors < 15.0) * 100
    
    print("\nEvaluation Results:")
    print(f"  Sample Size: {len(all_errors)}")
    print(f"  Mean Angular Error: {mean_error:.2f} degrees")
    print(f"  Std Angular Error:  {std_error:.2f} degrees")
    print("\nAccuracy (Percentage of samples within error threshold):")
    print(f"  < 5 degrees:  {acc_5:.1f}% (Excellent)")
    print(f"  < 10 degrees: {acc_10:.1f}% (Good)")
    print(f"  < 15 degrees: {acc_15:.1f}% (Acceptable)")

if __name__ == '__main__':
    evaluate()
