import random

import torch
import torch.nn as nn

from snakeMDP import Action



def epsilon_greedy(policy_network, state_world, epsilon):
    rand = random.random()
    if rand > epsilon:
        with torch.no_grad():
            state = torch.from_numpy(state_world)
            state = state.unsqueeze(0) # batch dim
            state = state.unsqueeze(0) # channel dim
            prediction = policy_network(state).max(1)[1].view(1, 1).item()
            action = Action(prediction)
    else:
        action = Action(random.randrange(4))
    return action


def softmax_policy(policy_network, state_world, temp):
    state = torch.from_numpy(state_world)
    state = state.unsqueeze(0) # batch dim
    state = state.unsqueeze(0) # channel dim
    with torch.no_grad():
        prediction = policy_network(state)[0]
        prob = nn.Softmax(dim=0)(prediction * (1.0/temp))
    rand = random.random()
    cum_prob = 0.0
    for i in range(len(prob)):
        cum_prob += prob[i]
        if rand < cum_prob:
            return Action(i)
    return Action(3)
