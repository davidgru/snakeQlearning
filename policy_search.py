from itertools import count

import torch
from torch.optim import Adam
from torch.distributions.categorical import Categorical


from dqn import CNN
from display import Display
from plot import GameStats, Plot
from snakeMDP import SnakeMDP
from eval import simulate


def play_game(snake, policy_network, discount, ttl = None, display = None):
    states = []
    actions = []
    rewards = []

    device = policy_network.device    
    state = snake.sample_start_state()
    stats = GameStats()
    score = 0
    _ttl = ttl

    while state:
        if display:
            display.draw(state.world, label='training')
            display.update()

        if ttl and _ttl <= 0:
            state = None
            break
        _ttl -= 1

        # need to convert numpy array to pytorch tensor and add batch and channel dimension
        state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0).to(device)

        # sample action according to policy
        with torch.no_grad():
            logits = policy_network(state_tensor)
            action_tensor = Categorical(logits=logits).sample()
            action = action_tensor.item()

        # add to trajectory
        states.append(state_tensor)
        actions.append(action_tensor)

        # obtain reward
        reward = snake.reward(state, action)
        rewards.append(reward * discount ** len(rewards))

        # get next state
        new_score, state = snake.next(state, action)
        stats.push(score)

        if ttl and new_score > score:
            _ttl = ttl
            score = new_score

    rewards = [sum(rewards)] * len(rewards)
    return states, actions, rewards, stats


def compile_batch(snake, policy_network, batch_size, discount, ttl = None, display = None):
    batch_states = []
    batch_actions = []
    batch_rewards = []

    statss = []

    while len(batch_states) < batch_size:
        states, actions, rewards, stats = play_game(snake, policy_network, discount, ttl, display)
        batch_states += states
        batch_actions += actions
        batch_rewards += rewards
        statss.append(stats)
    
    batch_states = torch.cat(batch_states)
    batch_actions = torch.cat(batch_actions)
    batch_rewards = torch.tensor(batch_rewards, device=policy_network.device)
    return batch_states, batch_actions, batch_rewards, statss
    

def compute_loss(policy_network, batch_states, batch_actions, batch_rewards):
    logits = policy_network(batch_states)
    logp = Categorical(logits=logits).log_prob(batch_actions)
    return -(logp * batch_rewards).mean()



def policy_search2(snake, policy_network, optimizer, batch_size = 512, discount = 0.95, ttl = None, cb = None, plot = None, display = None):
    
    for episode in count():
        batch_states, batch_actions, batch_rewards, statss = compile_batch(snake, policy_network, batch_size, discount, ttl, display)

        if plot:
            for stats in statss:
                plot.push(stats)

        optimizer.zero_grad()
        loss = compute_loss(policy_network, batch_states, batch_actions, batch_rewards)
        loss.backward()
        for param in policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if cb(episode, policy_network, optimizer) == True:
            return


def policy_search(snake, display, policy_network, optimizer, batch_size=512, discount=0.95, ttl=1000, cb=None):

    device = policy_network.device
    
    # collect batch_size samples
    for episode in count():

        batch_states = []
        batch_actions = []
        batch_rewards = []

        # play a game running the policy
        for game in count():

            state = snake.sample_start_state()
            stats = GameStats()
            score = 0
            _ttl = ttl

            game_rewards = []

            while state:

                display.draw(state.world, label='training')
                display.update()

                if _ttl <= 0:
                    state = None
                    break
                _ttl -= 1

                # need to convert numpy array to pytorch tensor and add batch and channel dimension
                state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0).to(device)

                # sample action according to policy
                with torch.no_grad():
                    logits = policy_network(state_tensor)
                    action_tensor = Categorical(logits=logits).sample()
                action = action_tensor.item()

                # add to trajectory
                batch_states.append(state_tensor)
                batch_actions.append(action_tensor)

                # obtain reward
                reward = snake.reward(state, action)
                game_rewards.append(reward * discount ** len(game_rewards))

                # get next state
                new_score, state = snake.next(state, action)
                stats.push(score)

                if new_score > score:
                    _ttl = ttl
                    score = new_score
                
            batch_rewards += [sum(game_rewards)] * len(game_rewards)
            
            if cb:
                cb(game, policy_network, optimizer, stats)

            if len(batch_states) >= batch_size:
                break



        state_tensor = torch.cat(batch_states)
        action_tensor = torch.cat(batch_actions)
        reward_tensor = torch.tensor(batch_rewards, device=device)

        optimizer.zero_grad()
        # compute loss        
        logits = policy_network(state_tensor)
        logp = Categorical(logits=logits).log_prob(action_tensor)
        loss = -(logp * reward_tensor).mean()
        loss.backward()
        
        # make optimizations step
        for param in policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


def main():

    WIDTH = 6
    HEIGHT = 6
    FADE = True

    TTL = 5 * WIDTH * HEIGHT

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    policy_network = CNN(HEIGHT, WIDTH, device).to(device)
    # display = Display(HEIGHT, WIDTH, 40)
    display = None
    snake = SnakeMDP(HEIGHT, WIDTH, 1, -1, -0.01, fade=FADE)

    plot = Plot(1000)

    optimizer = Adam(policy_network.parameters(), lr=0.002)

    def callback(episode, policy_network, optimizer):
        pass

    policy_search2(snake, policy_network, optimizer, batch_size=3000, discount=0.95, ttl=TTL, cb=callback, plot=plot, display=display)


if __name__ == '__main__':
    main()
