import random
import torch

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