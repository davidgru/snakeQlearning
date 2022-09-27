from email import policy
from time import sleep
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from display import Display
from dqn import DQN
from exploration_strategy import EpsilonGreedyPolicy
from replay_buffer import Transition, ReplayMemory
from snakeMDP import Action, SnakeMDP

HEIGHT = 10
WIDTH = 10
TILE_SIZE = 30

EXPLORATION_RATE = 0.05
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.01
TARGET_UPDATE_INTERVAL = 100
REPLAY_MEMORY_SIZE = 10000


# use qpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device.type)

# create snake mdp
snake = SnakeMDP(HEIGHT, WIDTH, 20.0, -1000.0, 0.1)

# create pygame display
display = Display(HEIGHT, WIDTH, TILE_SIZE)

# create policy and target networks
policy_network = DQN(HEIGHT, WIDTH, device).to(device)
target_network = DQN(HEIGHT, WIDTH, device).to(device)
target_network.load_state_dict(policy_network.state_dict())
target_network.eval()

optimizer = optim.RMSprop(policy_network.parameters(), lr=LEARNING_RATE)

replay_buffer = ReplayMemory(REPLAY_MEMORY_SIZE)
exploration_strategy = EpsilonGreedyPolicy(EXPLORATION_RATE)

state = snake.sample_start_state()

demo_interval = 1000

gameno = 0

for episode in count():

    display.draw(state.world)
    
    state_torch = torch.from_numpy(state.world)
    state_torch = state_torch.unsqueeze(0).unsqueeze(0)
    
    # sample action according to exploration strategy
    action = exploration_strategy.sample_action(policy_network, state.world)

    # compute temporal difference error
    reward = snake.reward(state, action)
    next_state = snake.next(state, action)

    # compute predicted Q value
    predicted_value = policy_network(state_torch)[0][action.value]
    # print(predicted_value.item())
    
    # compute target
    if next_state is not None:
        target_value = reward + target_network(state_torch).max(1)[0].view(1, 1).item()
        target_value = torch.tensor([target_value], device=device, dtype=torch.float32)
    else:
        next_state = snake.sample_start_state()
        target_value = torch.tensor([reward], device=device, dtype=torch.float32)
        gameno += 1

    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted_value.unsqueeze(0), target_value)
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_network.load_state_dict(policy_network.state_dict())

    state = next_state

    display.update()
    if gameno % demo_interval == 0:
        sleep(0.5)
