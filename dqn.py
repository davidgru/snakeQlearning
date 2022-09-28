import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):

    def __init__(self, height, width, device):
        super(DQN, self).__init__()
        
        self.height = height
        self.width = width
        self.device = device

        if self.height != 10 or self.width != 10:
            raise "Height and width are fixed for now!"

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.dense1 = nn.Linear(4*4*32, 4) # Output 4 Actions (left, right, up, down)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.dense1(x.view(x.size(0), -1))


class ResNet(nn.Module):

    def __init__(self, height, width, device):
        super(ResNet, self).__init__()
        
        self.height = height
        self.width = width
        self.device = device

        if self.height != 10 or self.width != 10:
            raise "Height and width are fixed for now!"

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        
        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.dense1 = nn.Linear(10*10*16, 4) # Output 4 Actions (left, right, up, down)

    def forward(self, x):
        x = x.to(self.device)

        x = F.relu(self.conv1(x))

        y = F.relu(self.conv21(x))
        y = F.relu(self.conv22(y))
        x = x + y

        y = F.relu(self.conv31(x))
        y = F.relu(self.conv32(y))
        x = x + y

        return self.dense1(x.view(x.size(0), -1))


