
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from time import sleep

class Plot:

    def __init__(self, window = 100):
        self.data = deque(maxlen=window)
        self.max = []
        self.avg = []
        self.min = []
        self.window_length = window
        plt.plot(self.max, label='Max', color='g')
        plt.plot(self.avg, label='Average', color='b')
        plt.plot(self.min, label='Min', color='r')
        plt.legend()
        plt.ion()
        plt.show()
        plt.pause(.1)

        self.points = 0
        self.last_update_len = 0

    def push(self, x):
        self.points += 1
        self.data.append(x)
        if self.points % self.window_length == 0:
            self.max.append(max(self.data))
            self.avg.append(sum(self.data) / self.window_length)
            self.min.append(min(self.data))

    def show(self):
        if self.last_update_len == len(self.avg):
            return
        self.last_update_len = len(self.avg)
        plt.plot(self.max, label='Max', color='g')
        plt.plot(self.avg, label='Average', color='b')
        plt.plot(self.min, label='Min', color='r')
        plt.pause(.01)
