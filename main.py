from email import policy
from select import epoll
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
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.0001
TARGET_UPDATE_INTERVAL = 10
REPLAY_MEMORY_SIZE = 10000


# use qpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device.type)

# create snake mdp
snake = SnakeMDP(HEIGHT, WIDTH, 1.0, 0.0, 0.0)

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

demo_interval = 500

gameno = 0
curr_time = 0
food = 0
surv_time = [0]
ttl = 1000

for episode in count():

    display.draw(state.world)
    
    state_torch = torch.from_numpy(state.world)
    state_torch = state_torch.unsqueeze(0).unsqueeze(0)
    
    # sample action according to exploration strategy
    action = exploration_strategy.sample_action(policy_network, state.world)

    # compute temporal difference error
    reward = snake.reward(state, action)
    if reward == 1.0:
        food += 1
        ttl = 1000
    next_state = snake.next(state, action)
    if ttl <= 0:
        next_state = None
    ttl -= 1

    # compute predicted Q value
    predicted_value = policy_network(state_torch)[0][action.value]
    # print(predicted_value.item())
    
    # compute target
    if next_state is not None:
        next_state_torch = torch.from_numpy(next_state.world)
        next_state_torch = next_state_torch.unsqueeze(0).unsqueeze(0)
        target_value = reward + target_network(next_state_torch).max(1)[0].view(1, 1).item()
        target_value = torch.tensor([target_value], device=device, dtype=torch.float32)
    else:
        next_state = snake.sample_start_state()
        target_value = torch.tensor([reward], device=device, dtype=torch.float32)
        gameno += 1
        surv_time.append(curr_time)
        curr_time = 0

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
        if curr_time == 0:
            print(f'time: {sum(surv_time) / len(surv_time):.2f}    food: {food / len(surv_time)}')
            surv_time.clear()
            food = 0
        if curr_time < 50:
            sleep(0.5)

    curr_time += 1
