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
TILE_SIZE = 40

EXPLORATION_RATE = 0.1
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
TARGET_UPDATE_INTERVAL = 50
REPLAY_MEMORY_SIZE = 1000000


# use qpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device.type)

# create snake mdp
snake = SnakeMDP(HEIGHT, WIDTH, 100.0, -1.0, -1.0)

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

demo_interval = 100

gameno = 1
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

    # step
    reward = snake.reward(state, action)
    next_state = snake.next(state, action)

    state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0)
    if next_state is not None:
        next_state_tensor = torch.from_numpy(next_state.world).unsqueeze(0).unsqueeze(0)
    else:
        next_state = snake.sample_start_state()
        next_state_tensor = None
        gameno += 1

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
    if gameno % demo_interval == 0:
        if curr_time == 0:
            print(f'time: {sum(surv_time) / len(surv_time):.2f}    food: {food / len(surv_time)}')
            surv_time.clear()
            food = 0
        if curr_time < 50:
            sleep(0.5)

    curr_time += 1
