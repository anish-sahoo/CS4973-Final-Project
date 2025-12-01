import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class MPIIGazeDataset(Dataset):
    """Dataset that provides left and right eye crops, optional head pose, and gaze labels.

    CSV format expected:
    image_path,l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2,pitch,yaw,head_pitch,head_yaw
    """
    def __init__(self, csv_file, transform=None, img_size=(36,60)):
        self.data = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['image_path']).convert('L')
        # crop left eye
        lx1, ly1, lx2, ly2 = int(row['l_x1']), int(row['l_y1']), int(row['l_x2']), int(row['l_y2'])
        rx1, ry1, rx2, ry2 = int(row['r_x1']), int(row['r_y1']), int(row['r_x2']), int(row['r_y2'])
        left = img.crop((lx1, ly1, lx2, ly2))
        right = img.crop((rx1, ry1, rx2, ry2))
        left = self.transform(left)
        right = self.transform(right)
        gaze = torch.tensor([row['pitch'], row['yaw']], dtype=torch.float32)
        head = torch.tensor([row['head_pitch'], row['head_yaw']], dtype=torch.float32)
        return left, right, head, gaze
