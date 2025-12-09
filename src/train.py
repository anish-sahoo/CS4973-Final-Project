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
from datasets.mpiigaze_dataset import MPIIGazeDataset, MPIIGazeNormalizedDataset
from models.gaze_model import GazeNet
from visualizer import TrainingVisualizer
from utils import angular_error_loss


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
    valid_batches = 0
    
    # Get gradient clipping config
    grad_clip = getattr(config, 'GRAD_CLIP_MAX_NORM', None)
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (left_imgs, right_imgs, head, gaze) in progress_bar:
        left_imgs = left_imgs.to(device, non_blocking=True)
        right_imgs = right_imgs.to(device, non_blocking=True)
        head = head.to(device, non_blocking=True)
        gaze = gaze.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision context
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(left_imgs, right_imgs, head)
            loss = criterion(pred, gaze)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWarning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Scale loss and backward
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        valid_batches += 1
        
        # Log to tensorboard at intervals
        step = epoch * num_batches + batch_idx
        # Log to visualizer every 100 steps to reduce storage
        if step % 100 == 0:
            visualizer.log_train_loss(batch_loss, epoch, step=step)
        # Update progress bar at configured interval
        if (batch_idx + 1) % log_interval == 0:
            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    avg_loss = total_loss / max(valid_batches, 1)
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
    
    # Enable cuDNN benchmark for optimized performance on fixed input sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(log_dir=config.LOG_DIR, tb_enabled=use_tensorboard)
    
    # Load dataset
    if getattr(config, 'USE_NORMALIZED', False):
        print(f"Loading normalized dataset from {config.NORMALIZED_ROOT}...")
        dataset = MPIIGazeNormalizedDataset(config.NORMALIZED_ROOT)
    else:
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
    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Initialize model
    print("Initializing model...")
    model = GazeNet().to(device)
    
    # Select optimizer based on config
    optimizer_type = getattr(config, 'OPTIMIZER', 'adam').lower()
    weight_decay = getattr(config, 'WEIGHT_DECAY', 0.0)
    
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=weight_decay)
        optimizer_name = f'AdamW (lr={config.LR}, wd={weight_decay})'
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=0.9, weight_decay=weight_decay)
        optimizer_name = f'SGD (lr={config.LR}, momentum=0.9, wd={weight_decay})'
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=weight_decay)
        optimizer_name = f'Adam (lr={config.LR}, wd={weight_decay})'
    
    # Setup learning rate scheduler
    scheduler_type = getattr(config, 'LR_SCHEDULER', 'none').lower()
    scheduler = None
    scheduler_name = 'None'
    
    if scheduler_type == 'cosine':
        lr_min = getattr(config, 'LR_MIN', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)
        scheduler_name = f'CosineAnnealing (min_lr={lr_min})'
    elif scheduler_type == 'step':
        step_size = getattr(config, 'LR_STEP_SIZE', 10)
        gamma = getattr(config, 'LR_GAMMA', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_name = f'StepLR (step={step_size}, gamma={gamma})'
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scheduler_name = 'ReduceLROnPlateau (factor=0.5, patience=5)'
    
    # Select loss function based on config
    loss_fn_type = getattr(config, 'LOSS_FUNCTION', 'mse').lower()
    if loss_fn_type == 'angular':
        criterion = angular_error_loss
        criterion_name = 'Angular Error Loss'
    else:
        criterion = nn.MSELoss()
        criterion_name = 'MSE Loss'
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer_name}")
    print(f"LR Scheduler: {scheduler_name}")
    print(f"Criterion: {criterion_name}")
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
        
        # Step the learning rate scheduler
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if use_tensorboard and visualizer.writer:
                visualizer.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}", end='')
        if scheduler is not None:
            print(f" - LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print()
            
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
