import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from src.datasets.mpiigaze_dataset import MPIIGazeDataset
import config
from src.models.gaze_model import GazeNet
import torch.nn as nn
import torch.optim as optim

dataset = MPIIGazeDataset(config.CSV_PATH)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
#loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Track best model
best_val_loss = float('inf')

for epoch in range(10):  # More epochs
    model.train()
    train_loss = 0
    for left_imgs, right_imgs, head, gaze in train_loader:
        left_imgs = left_imgs.to(device)
        right_imgs = right_imgs.to(device)
        head = head.to(device)
        gaze = gaze.to(device)
        optimizer.zero_grad()
        pred = model(left_imgs, right_imgs, head)
        loss = criterion(pred, gaze)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for left_imgs, right_imgs, head, gaze in val_loader:
            left_imgs, right_imgs, head, gaze = left_imgs.to(device), right_imgs.to(device), head.to(device), gaze.to(device)
            pred = model(left_imgs, right_imgs, head)
            loss = criterion(pred, gaze)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "gaze_two_eye_best.pth")
    
    print(f"Epoch {epoch+1} Train Loss {train_loss/len(train_loader):.4f} Val Loss {val_loss:.4f}")

torch.save(model.state_dict(), "gaze_two_eye.pth")
