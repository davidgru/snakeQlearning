from itertools import count
import math

import torch
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from dqn import CNN
from display import Display
from exploration_strategy import softmax_policy
from plot import GameStats, Plot
from replay_buffer import ReplayMemory, Transition
from snakeMDP import SnakeMDP



def optimize(policy_network, target_network, optimizer, batch, discount_factor):
    device = policy_network.device
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
    
    # compute predicted q values
    q_values = policy_network(state_batch).gather(1, action_batch)
    
    # compute value of next state if non-final
    next_state_values = torch.zeros(state_batch.shape[0], device=device)
    next_state_values[non_final_mask] = target_network(next_state_batch).max(1)[0].detach()
    
    # compute expected q_values
    expected_q_values = (next_state_values * discount_factor) + reward_batch
    
    # compute loss and make step
    criterion = SmoothL1Loss()
    loss = criterion(q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def play_game(episode, snake, policy_network, target_network, replay_buffer, optimizer, tmp, batch_size, discount_factor, target_update, ttl = None, display = None):
    device = policy_network.device

    state = snake.sample_start_state()
    stats = GameStats()
    score = 0
    _ttl = ttl

    while state:

        if display:
            display.draw(state.world, 'training')
            display.update()

        if ttl and _ttl <= 0:
            state = None
            break
        _ttl -= 1

        # need to convert numpy world to torch tensor and add dimensions for channel and batch
        state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0).to(device)
        
        # sample action according to exploration strategy
        action = softmax_policy(policy_network, state.world, tmp)

        # advance the environment and get reward
        reward = snake.reward(state, action)
        new_score, next_state  = snake.next(state, action)
        stats.push(new_score)
        
        if new_score > score:
            score = new_score
            ttl = _ttl


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
        if len(replay_buffer) >= batch_size:
            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))
            optimize(policy_network, target_network, optimizer, batch, discount_factor)

        # update target network
        if episode % target_update == 0:
            target_network.load_state_dict(policy_network.state_dict())

        state = next_state
        episode += 1

    return episode, stats


def deep_q_learning(snake, policy_network, target_network, optimizer, tmp_start, tmp_end, tmp_decay, batch_size, discount_factor, target_update, replay_capacity, ttl = None, plot = None, display = None, cb = None):
    replay_buffer = ReplayMemory(replay_capacity)
    episode = 0
    for game in count():
        tmp = tmp_end + (tmp_start - tmp_end) * math.exp(-1. * game / tmp_decay)
        episode, stats = play_game(episode, snake, policy_network, target_network, replay_buffer, optimizer, tmp, batch_size, discount_factor, target_update, ttl, display)
        plot.push(stats)

        if cb:
            cb(episode, game, policy_network, optimizer)


def main():
    
    WIDTH = 6
    HEIGHT = 6
    FADE = True

    FOOD_REWARD = 1.0
    LIVING_REWARD = -0.01
    DEATH_REWARD = -1.0

    DISCOUNT_FACTOR = 0.95
    TTL = 10 * WIDTH * HEIGHT
    TARGET_UPDATE_INTERVAL = 500
    REPLAY_MEMORY_SIZE = 500000

    # parameters for softmax temperature
    TMP_START = 100.0
    TMP_END = 0.01
    TMP_DECAY = 700

    LEARNING_RATE = 0.001
    BATCH_SIZE = 512

    SAVE_INTERVAL = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snake = SnakeMDP(HEIGHT, WIDTH, FOOD_REWARD, DEATH_REWARD, LIVING_REWARD, fade=FADE)
    display = Display(HEIGHT, WIDTH, 40)

    # create policy and target networks
    policy_network = CNN(HEIGHT, WIDTH, device).to(device)
    target_network = CNN(HEIGHT, WIDTH, device).to(device)
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    optimizer = Adam(policy_network.parameters(), lr=LEARNING_RATE)

    plot = Plot(20)

    def cb(_, game, policy_network, optimizer):
        if game % SAVE_INTERVAL:
            torch.save({
                'width': WIDTH,
                'height': HEIGHT,
                'fade': FADE,
                'state_dict': policy_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'game': game,
            }, "./my_model.pt")

    deep_q_learning(snake, policy_network, target_network, optimizer, TMP_START, TMP_END, TMP_DECAY, BATCH_SIZE, DISCOUNT_FACTOR, TARGET_UPDATE_INTERVAL, REPLAY_MEMORY_SIZE, TTL, plot, display, cb)


if __name__ == '__main__':
    main()
