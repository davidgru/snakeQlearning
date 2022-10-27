# Proximal Policy Optimization
# Credit: https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb

from itertools import count

import numpy as np
import os
import torch
from torch.distributions import Categorical
from torch.optim import Adam

from actorcritic_model import ActorCriticModel
from display import Display
from plot import GameStats, Plot
from save import save
from snakeMDP import Action, SnakeMDP

# generalized advantage estimation
def gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.append(gae + values[step])
    returns.reverse()
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(model, optimizer, epochs, mini_batch_size, states, actions, log_probs, returns, advantages, epsilon_clip=0.2):
    for _ in range(epochs):
        for state, action, old_log_probs, return_, advantage, in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            logits, value = model(state)
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss =  (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def ppo(snake, model, device, optimizer, num_steps, epochs, mini_batch_size, gamma, tau, epsilon_clip, ttl = None, plot = None, display = None, cb = None):
    state = snake.sample_start_state()
    game = 0
    score = 0
    stats = GameStats()
    if ttl:
        _ttl = ttl

    for episode in count():

        log_probs, values, states, actions, rewards, masks, entropy = [], [], [], [], [], [], 0

        for _ in range(num_steps):
            
            if display:
                display.draw(state.world, "training")
                display.update()

            state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0).to(device)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()
            action = Action(action_tensor.item())

            reward = snake.reward(state, action)
            new_score, next_state = snake.next(state, action)
            stats.push(new_score)

            if new_score > score:
                _ttl = ttl
                score = new_score

            if not next_state or ttl and _ttl <= 0:
                next_state = snake.sample_start_state()
                plot.push(stats)
                score = 0
                stats = GameStats()
                done = True
                _ttl = ttl
                game += 1
                if cb:
                    cb(game, model, optimizer)
            else:
                done = False
                _ttl -= 1

            log_prob = dist.log_prob(action_tensor)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob.unsqueeze(0))
            values.append(value)
            rewards.append(torch.tensor([reward], device=device))
            masks.append(torch.tensor([1 - int(done)], device=device))
            
            states.append(state_tensor)
            actions.append(action_tensor.unsqueeze(0))

            state = next_state

        next_state = torch.from_numpy(next_state.world).unsqueeze(0).unsqueeze(0).to(device)
        _, next_value = model(next_state)
        returns = gae(next_value, rewards, masks, values, gamma, tau)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states).detach()
        actions = torch.cat(actions).detach()
        advantages = returns - values

        ppo_update(model, optimizer, epochs, mini_batch_size, states, actions, log_probs, returns, advantages, epsilon_clip)


def main():
    
    HEIGHT = 10
    WIDTH = 10
    FADE = True

    # rewards
    FOOD_REWARD = 1.0
    LIVING_REWARD = 0.0
    DEATH_REWARD = 0.0

    # Hyperparams
    LR = 0.00003
    NUM_STEPS = 64
    MINIBATCH_SIZE = 1
    EPOCHS = 2
    GAMMA = 0.95 # discount factor
    TAU = 0.95 # TODO: What is it?
    EPSILON = 0.2 # ppo clip threshold
    TTL = 5 * HEIGHT * WIDTH

    USE_DISPLAY = False
    PLOT_GRANULARITY = 100
    SAVE_DIR = 'save'
    SAVE_INTERVAL = 1000

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = ActorCriticModel(HEIGHT, WIDTH, FADE, kernel=5, channels=64, depth=3, hidden=256).to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    snake = SnakeMDP(HEIGHT, WIDTH, FOOD_REWARD, DEATH_REWARD, LIVING_REWARD, fade=FADE)
    display = Display(HEIGHT, WIDTH, 40) if USE_DISPLAY else None
    display = None
    plot = Plot(PLOT_GRANULARITY)

    def callback(game, model, optimizer):
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if game % SAVE_INTERVAL == 0:
            save(model, os.path.join(SAVE_DIR, f'{game}.pt'))


    ppo(snake, model, device, optimizer, NUM_STEPS, EPOCHS, MINIBATCH_SIZE, GAMMA, TAU, EPSILON, TTL, plot, display, callback)


if __name__ == '__main__':
    main()
