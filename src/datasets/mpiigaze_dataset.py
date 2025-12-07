import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np

class MPIIGazeDataset(Dataset):
    """Dataset that provides left and right eye crops, optional head pose, and gaze labels.
    Optimized with OpenCV and pre-converted data structures.

    CSV format expected:
    image_path,l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2,pitch,yaw,head_pitch,head_yaw
    """
    def __init__(self, csv_file, transform=None, img_size=(36,60)):
        df = pd.read_csv(csv_file)
        self.data = df.to_dict('records')
        self.img_size = img_size # (height, width)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not read image: {row['image_path']}")

        # Crop coordinates
        lx1, ly1, lx2, ly2 = int(row['l_x1']), int(row['l_y1']), int(row['l_x2']), int(row['l_y2'])
        rx1, ry1, rx2, ry2 = int(row['r_x1']), int(row['r_y1']), int(row['r_x2']), int(row['r_y2'])
        
        # Crop eyes
        left = img[ly1:ly2, lx1:lx2]
        right = img[ry1:ry2, rx1:rx2]
        
        target_w, target_h = self.img_size[1], self.img_size[0]
        left = cv2.resize(left, (target_w, target_h))
        right = cv2.resize(right, (target_w, target_h))
        
        left = left.astype(np.float32) / 255.0
        right = right.astype(np.float32) / 255.0
        
        # (x - mean) / std with mean=0.5, std=0.5 => (x - 0.5) / 0.5
        left = (left - 0.5) / 0.5
        right = (right - 0.5) / 0.5
        
        # (H, W) -> (1, H, W)
        left = torch.from_numpy(left).unsqueeze(0)
        right = torch.from_numpy(right).unsqueeze(0)
        
        gaze = torch.tensor([row['pitch'], row['yaw']], dtype=torch.float32)
        head = torch.tensor([row['head_pitch'], row['head_yaw']], dtype=torch.float32)
        return left, right, head, gaze
