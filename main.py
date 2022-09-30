from time import sleep, time
from itertools import count
import math

import torch
import torch.nn as nn
import torch.optim as optim

from display import Display
from dqn import DQN, ResNet
from exploration_strategy import epsilon_greedy, softmax_policy
from hyperparameters import Hyperparameters
from replay_buffer import Transition, ReplayMemory
from snakeMDP import Action, SnakeMDP
from plot import Plot

HEIGHT = 10
WIDTH = 10
TILE_SIZE = 40

TEMP_START = 100.0
TEMP_END = 0.01
TEMP_DECAY = 850
DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.0005
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL = 500
REPLAY_MEMORY_SIZE = 1000000

FOOD_REWARD = 1
DEATH_REWARD = -2
LIVING_REWARD = -0.01

temp = TEMP_START

# hyperparameters are changeable at runtime via commandline
# example: \>set exploration_rate 0.2
hyperparams = Hyperparameters(exploration_rate=TEMP_START, discount_factor=DISCOUNT_FACTOR, update_interval=TARGET_UPDATE_INTERVAL, slow='False')


# use qpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create snake mdp
snake = SnakeMDP(HEIGHT, WIDTH, FOOD_REWARD, DEATH_REWARD, LIVING_REWARD)

# create pygame display
display = Display(HEIGHT, WIDTH, TILE_SIZE)

# create policy and target networks
policy_network = ResNet(HEIGHT, WIDTH, device).to(device)
target_network = ResNet(HEIGHT, WIDTH, device).to(device)
target_network.load_state_dict(policy_network.state_dict())
target_network.eval()

optimizer = optim.RMSprop(policy_network.parameters(), lr=LEARNING_RATE)

replay_buffer = ReplayMemory(REPLAY_MEMORY_SIZE)

state = snake.sample_start_state()

gameno = 1

plot = Plot(20)

food = 0

last_plot = time()

for episode in count():

    # print(hyperparams.get_state())

    display.draw(state.world)
    
    state_torch = torch.from_numpy(state.world)
    state_torch = state_torch.unsqueeze(0).unsqueeze(0)
    
    # sample action according to exploration strategy
    action = softmax_policy(policy_network, state.world, temp)
    temp = TEMP_END + (TEMP_START - TEMP_END) * \
        math.exp(-1. * gameno / TEMP_DECAY)

    # step
    reward = snake.reward(state, action)
    next_state = snake.next(state, action)

    if reward == FOOD_REWARD:
        food += 1

    state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0)
    if next_state is not None:
        next_state_tensor = torch.from_numpy(next_state.world).unsqueeze(0).unsqueeze(0)
    else:
        next_state = snake.sample_start_state()
        next_state_tensor = None
        plot.push(food)
        food = 0
        gameno += 1
        print(temp)

    replay_buffer.push(Transition(state_tensor, torch.tensor([[action.value]], device=device), next_state_tensor, torch.tensor([reward], device=device)))

    # optimize
    if len(replay_buffer) >= BATCH_SIZE:
        transitions = replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_network(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()#
        expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_network.load_state_dict(policy_network.state_dict())

    state = next_state


    display.update()

    if time() - last_plot >= 10:
        last_plot = time()
        plot.show()


    if (hyperparams['slow'] == 'True'):
        sleep(0.5)
