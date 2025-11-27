"""
A small, lightweight CNN to predict gaze coordinates (x,y normalized) from an eye image.
"""

import torch
import torch.nn as nn


class SimpleEyeGazeNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # output gaze x,y
            nn.Sigmoid(),  # normalized between 0 and 1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


def build_model() -> SimpleEyeGazeNet:
    return SimpleEyeGazeNet()
