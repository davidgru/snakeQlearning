import sys
from statistics import median
from time import sleep


import torch

from display import Display
from dqn import CNN
from plot import GameStats
from snakeMDP import Action, SnakeMDP
from exploration_strategy import softmax_policy


def test(policy_network, snake, display = None, delay = 0.0, ttl = 2000):

    state = snake.sample_start_state()
    stats = GameStats()
    score = 0

    ttll = ttl

    while state is not None:

        if display:
            display.draw(state.world, "testing")
            display.update()
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0)
        action = softmax_policy(policy_network, state.world, 0.00001)
        # action = Action(policy_network(state_tensor).max(1)[1].view(1, 1).item())

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

    device = torch.device('cpu')
    info = torch.load(argv[1], map_location=device)

    height = info['height']
    width = info['width']
    fade = info['fade']


    model = CNN(height, width, device).to(device)
    model.load_state_dict(info['state_dict'])
    model.eval()

    snake = SnakeMDP(height, width, 0, 0, 0, fade=fade)
    
    display = Display(height, width, 40)

    test(model, snake, display, delay=0.25, ttl=2*height*width)

if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
