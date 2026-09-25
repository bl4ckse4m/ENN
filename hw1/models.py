import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes = 100):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias = False
        )

        self.pool = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2,
            bias=False
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            padding=0,
            bias=False
        )

        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride = 2,
            padding=1,
            bias=False
        )

        self.conv6 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            padding=0,
            bias=False
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(
            512,
            256,
        )

        self.fc2 = nn.Linear(
            256,
            num_classes,
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))

        return self.fc2(x)