import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        # input image size used elsewhere: (1,36,60) -> after 3 poolings? 36x60 -> pools: 18x30, 9x15, then conv layer: we want final features of 128*9*15? We'll use flatten and a small FC
        # assuming input images are (1, 36, 60): after 3 maxpools (2x2) we get (4,7)
        self.fc = nn.Linear(128*4*7, 128)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class GazeNet(nn.Module):
    def __init__(self, head_feat=32):
        super().__init__()
        self.left_eye = EyeCNN()
        self.right_eye = EyeCNN()
        self.head_fc = nn.Linear(2, head_feat)
        self.fc1 = nn.Linear(128*2 + head_feat, 256)
        self.fc2 = nn.Linear(256, 2)  # output pitch & yaw

    def forward(self, left, right, head):
        # left, right: Bx1xH x W, head: Bx2
        lf = self.left_eye(left)
        rf = self.right_eye(right)
        hf = F.relu(self.head_fc(head))
        x = torch.cat([lf, rf, hf], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GazeCNN(nn.Module):
    # Keep a single-eye model for backward compatibility
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128*4*6, 256)
        self.fc2 = nn.Linear(256, 2)  # pitch & yaw

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
