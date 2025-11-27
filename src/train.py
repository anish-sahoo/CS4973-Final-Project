import torch
from torch.utils.data import DataLoader
from datasets.mpiigaze_dataset import MPIIGazeDataset
from models.gaze_model import GazeCNN
import torch.nn as nn
import torch.optim as optim

dataset = MPIIGazeDataset("mpiigaze_train.csv")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for imgs, gaze in loader:
        imgs, gaze = imgs.to(device), gaze.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, gaze)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "gaze_cnn.pth")
