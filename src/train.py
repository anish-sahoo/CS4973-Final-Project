import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from datasets.mpiigaze_dataset import MPIIGazeDataset
from models.gaze_model import GazeNet
from visualizer import TrainingVisualizer


def get_device():
    """Determine the best device to use."""
    if config.DEVICE == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(config.DEVICE)
    return device


def setup_directories():
    """Create necessary directories for outputs."""
    Path(config.LOG_DIR).mkdir(exist_ok=True)
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, visualizer, log_interval):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (left_imgs, right_imgs, head, gaze) in progress_bar:
        left_imgs = left_imgs.to(device)
        right_imgs = right_imgs.to(device)
        head = head.to(device)
        gaze = gaze.to(device)
        
        optimizer.zero_grad()
        pred = model(left_imgs, right_imgs, head)
        loss = criterion(pred, gaze)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        
        # Log to tensorboard at intervals
        step = epoch * num_batches + batch_idx
        if (batch_idx + 1) % log_interval == 0:
            visualizer.log_train_loss(batch_loss, epoch, step=step)
            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train():
    """
    Train the gaze tracking model.
    Configuration is loaded from config.py
    """
    mode = config.TRAIN_MODE
    use_tensorboard = config.USE_TENSORBOARD
    
    print(f"Starting training in '{mode}' mode")
    print(f"Device: {get_device()}")
    
    # Setup
    device = get_device()
    setup_directories()
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(log_dir=config.LOG_DIR, tb_enabled=use_tensorboard)
    
    # Load dataset
    print(f"Loading dataset from {config.CSV_PATH}...")
    dataset = MPIIGazeDataset(config.CSV_PATH)
    
    # Determine dataset size based on mode
    if mode == 'short':
        # Use only first 100 samples for quick testing
        num_samples = min(100, len(dataset))
        dataset = Subset(dataset, list(range(num_samples)))
        num_epochs = 2
        print(f"Short mode: Training on {num_samples} samples for {num_epochs} epochs")
    elif mode == 'complete':
        num_epochs = config.EPOCHS
        print(f"Complete mode: Training on {len(dataset)} samples for {num_epochs} epochs")
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'short' or 'complete'.")
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Initialize model
    print("Initializing model...")
    model = GazeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.MSELoss()
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: Adam (lr={config.LR})")
    print(f"Criterion: MSELoss")
    print(f"Batch size: {config.BATCH_SIZE}")
    print()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            model, loader, optimizer, criterion, device, 
            epoch, visualizer, config.LOG_INTERVAL
        )
        visualizer.log_epoch(epoch, avg_loss)
        if (epoch + 1) % config.PLOT_SAVE_INTERVAL == 0 or (epoch + 1) == num_epochs:
            visualizer.plot()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'gaze_best_{mode}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")
    
    model_path = os.path.join(config.CHECKPOINT_DIR, f'gaze_final_{mode}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved to {model_path}")
    
    summary = visualizer.get_summary()
    print("\nTraining Summary:")
    print(f"  Total epochs: {summary['total_epochs']}")
    print(f"  Final train loss: {summary['final_train_loss']:.4f}")
    print(f"  Best train loss: {summary['best_train_loss']:.4f}")
    
    visualizer.flush()
    visualizer.close()
    
    return model, visualizer


if __name__ == '__main__':
    train()
