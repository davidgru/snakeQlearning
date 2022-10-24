
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticModel(nn.Module):

    def __init__(self, height, width, fade, kernel=5, channels=64, depth=3, hidden=256, std=0.0):
        super(ActorCriticModel, self).__init__()

        self.height = height
        self.width = width
        self.fade = fade

        padding = (kernel - 1) // 2
        stride = 1

        self.convs = [nn.Conv2d(1, channels, kernel_size=kernel, stride=stride, padding=padding)] \
            + [nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=padding) for _ in range(depth-1)]
        self.convs = nn.ModuleList(self.convs)
        self.dense = nn.Linear(height*width*channels, hidden)
        self.critic = nn.Linear(hidden, 1)
        self.actor = nn.Linear(hidden, 4)


    @torch.jit.export
    def info(self):
        return {
            'height': self.height,
            'width': self.width,
            'fade': int(self.fade),
            'critic': int(True)
        }


    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.dense(x.view(x.size(0), -1))
        value = self.critic(x)
        logits = self.actor(x)
        return logits, value
