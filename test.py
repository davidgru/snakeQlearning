import sys
from statistics import median
from time import sleep


import torch
from torch.distributions import Categorical

from display import Display
from plot import GameStats
from snakeMDP import Action, SnakeMDP
from exploration_strategy import softmax_policy


@torch.no_grad()
def test(policy_network, device, snake, display = None, delay = 0.0, ttl = 2000):

    state = snake.sample_start_state()
    stats = GameStats()
    score = 0

    ttll = ttl

    while state is not None:

        if display:
            display.draw(state.world, "testing")
            display.update()
        
        state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0).to(device)
        
        if policy_network.info()['critic']:
            logits, _ = policy_network(state_tensor)
            dist = Categorical(logits=logits)
            action = Action(dist.sample().item())
        else:
            action = softmax_policy(policy_network, state.world, 0.00001)

        new_score, state = snake.next(state, action)
        stats.push(new_score)

        if new_score > score:
            ttll = ttl
        score = new_score

        ttll -= 1
        if ttll <= 0:
            return stats
        
        sleep(delay)

    return stats


def main(argc, argv):
    if argc < 2:
        sys.exit(1)

    use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = torch.jit.load(argv[1]).to(device)
    model.eval()

    info = model.info()

    height = info['height']
    width = info['width']
    fade = bool(info['fade'])

    snake = SnakeMDP(height, width, 0, 0, 0, fade=fade)
    
    display = Display(height, width, 40)

    test(model, device, snake, display, delay=0.25, ttl=2*height*width)


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
