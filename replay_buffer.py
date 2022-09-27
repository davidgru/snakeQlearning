from collections import deque, namedtuple
import random


Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])


class ReplayMemory:

    def __init__(self, capacity):
        self.data = deque([], maxlen=capacity)

    def push(self, transition):
        self.data.append(transition)

    def sample(self, count):
        return random.sample(self.data, count)

    def __len__(self):
        return len(self.data)
