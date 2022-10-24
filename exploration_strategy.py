import random

import torch
import torch.nn.functional as F

from snakeMDP import Action


@torch.no_grad()
def epsilon_greedy(policy_network, state_world, epsilon):
    rand = random.random()
    if rand > epsilon:
        state = torch.from_numpy(state_world).unsqueeze(0).unsqueeze(0)
        prediction = policy_network(state).max(1)[1].view(1, 1).item()
        action = Action(prediction)
    else:
        action = Action(random.randrange(4))
    return action


@torch.no_grad()
def softmax_policy(policy_network, state_world, temp):
    device = policy_network.device

    state = torch.from_numpy(state_world).unsqueeze(0).unsqueeze(0)
    prediction = policy_network(state)[0]
    prob = F.softmax(prediction * (1.0/temp), dim=0)
    rand = random.random()
    cum_prob = 0.0
    for i in range(len(prob)):
        cum_prob += prob[i]
        if rand < cum_prob:
            return Action(i)
    return Action(3)
