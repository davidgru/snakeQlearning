import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, height, width, device):
        super(CNN, self).__init__()
        
        self.height = height
        self.width = width
        self.device = device

        if self.height <= 3 or self.width <= 3:
            raise "Too small"

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.dense1 = nn.Linear(height*width*16, 4)


    def forward(self, x):
        x = x.to(self.device)

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        
        return self.dense1(x.view(x.size(0), -1))

