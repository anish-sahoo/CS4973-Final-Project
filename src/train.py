import torch
from torch.utils.data import DataLoader
from datasets.mpiigaze_dataset import MPIIGazeDataset
import config
from models.gaze_model import GazeNet
import torch.nn as nn
import torch.optim as optim

dataset = MPIIGazeDataset(config.CSV_PATH)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for left_imgs, right_imgs, head, gaze in loader:
        left_imgs = left_imgs.to(device)
        right_imgs = right_imgs.to(device)
        head = head.to(device)
        gaze = gaze.to(device)
        optimizer.zero_grad()
        pred = model(left_imgs, right_imgs, head)
        loss = criterion(pred, gaze)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "gaze_two_eye.pth")
