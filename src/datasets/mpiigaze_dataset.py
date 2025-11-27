import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class MPIIGazeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((36, 60)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['eye_path']).convert("L")  # grayscale
        img = self.transform(img)
        gaze = torch.tensor([row['pitch'], row['yaw']], dtype=torch.float32)
        return img, gaze
