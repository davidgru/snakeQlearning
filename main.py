from time import sleep, time
from itertools import count
import math

import torch
import torch.nn as nn
import torch.optim as optim

from display import Display
from dqn import CNN
from exploration_strategy import epsilon_greedy, softmax_policy
from plot import GameStats, Plot
from replay_buffer import Transition, ReplayMemory
from snakeMDP import SnakeMDP

HEIGHT = 6
WIDTH = 6
TILE_SIZE = 40

FADE = True
TTL = 10 * HEIGHT * WIDTH

TEMP_START = 100.0
TEMP_END = 0.01
TEMP_DECAY = 500
DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL = 500
REPLAY_MEMORY_SIZE = 500000

FOOD_REWARD = 1
DEATH_REWARD = -3
LIVING_REWARD = -0.01

PLOT_GRANULARITY = 20
SAVE_INTERVAL = 200

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create snake mdp
snake = SnakeMDP(HEIGHT, WIDTH, FOOD_REWARD, DEATH_REWARD, LIVING_REWARD, fade=FADE)

# create pygame display
display = Display(HEIGHT, WIDTH, TILE_SIZE)

# create policy and target networks
policy_network = CNN(HEIGHT, WIDTH, device).to(device)
target_network = CNN(HEIGHT, WIDTH, device).to(device)
target_network.load_state_dict(policy_network.state_dict())
target_network.eval()

# Create replay memory
replay_buffer = ReplayMemory(REPLAY_MEMORY_SIZE)

optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)


plot = Plot(PLOT_GRANULARITY)

episode = 0
for game in count():

    state = snake.sample_start_state()
    stats = GameStats()
    score = 0
    ttl = TTL

    while state:

        display.draw(state.world, "training")
        display.update()

        # need to convert numpy world to torch tensor and add dimensions for channel and batch
        state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0).to(device)
        
        # sample action according to exploration strategy
        temp = TEMP_END + (TEMP_START - TEMP_END) * math.exp(-1. * game / TEMP_DECAY)
        action = softmax_policy(policy_network, state.world, temp)

        # advance the environment and get reward
        reward = snake.reward(state, action)
        new_score, next_state  = snake.next(state, action)
        stats.push(new_score)
        
        # stop game after certain amount of steps without progress
        if new_score > score:
            score = new_score
            ttl = TTL

        if ttl <= 0:
            reward = DEATH_REWARD
            next_state = None
        ttl -= 1

        # need convert all components of transition to torch tensors
        if next_state:
            next_state_tensor = torch.from_numpy(next_state.world).unsqueeze(0).unsqueeze(0).to(device)
        else:
            next_state_tensor = None
        action_tensor = torch.tensor([[action]], device=device)
        reward_tensor = torch.tensor([reward], device=device)
        transition = Transition(state_tensor, action_tensor, next_state_tensor, reward_tensor)
        replay_buffer.push(transition)

        # make optimization step
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

        # update target network
        if episode % TARGET_UPDATE_INTERVAL == 0:
            target_network.load_state_dict(policy_network.state_dict())

        state = next_state
        episode += 1

    plot.push(stats)

    if game % SAVE_INTERVAL:
        torch.save({
            'width': WIDTH,
            'height': HEIGHT,
            'fade': FADE,
            'state_dict': policy_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'game': game,
        }, "./my_model.pt")
