import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 1)
        self.conv5 = nn.Conv2d(128, 192, 5, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv6 = nn.Conv2d(192, 256, 6, padding=1)

        self.fc1 = nn.Linear(250*2*2, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))))
        x = self.pool2(F.relu(self.conv5(F.relu(self.conv4(x)))))
        x = F.relu(self.conv6(x))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x