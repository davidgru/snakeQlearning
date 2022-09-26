import random
import torch


class EpsilonGreedyPolicy:

    def __init__(self, epsilon = 0.1, decay = 0.0):
        self.epsilon = epsilon
        self.decay = decay

    def sample_action(self, policy_network, state):
        rand = random.random()
        if rand > self.epsilon:
            with torch.no_grad():
                return policy_network(state).max(1)[1].view(1, 1)[0]
        else:
            return torch.tensor([random.randrange(4)], device=policy_network.device(), dtype=torch.long)
