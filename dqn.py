import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, height, width, fade, device):
        super(CNN, self).__init__()
        
        self.height = height
        self.width = width
        self.fade = fade
        self.device = device

        if self.height < 3 or self.width < 3:
            raise "Too small"

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.dense1 = nn.Linear(height*width*64, 256)
        self.dense2 = nn.Linear(256, 4)


    @torch.jit.export
    def info(self):
        return {
            'height': self.height,
            'width': self.width,
            'fade': int(self.fade),
            'critic': int(False)
        }


    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dense1(x.view(x.size(0), -1))
        return self.dense2(x)
