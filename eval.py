import matplotlib.pyplot as plt
from statistics import median
import sys
import torch
import numpy as np

from snakeMDP import SnakeMDP
from test import test

# simulates specified number of games and returns a list of scores
def simulate(policy_network, device, snake, iterations=0, ttl=2000, cb=None, display=None):
    scores = []
    for i in range(iterations):
        stats = test(policy_network, device, snake, ttl=ttl, display=display)
        scores.append(stats.score)
        if cb:
            cb(i, stats.score)
    return scores


def main(argc, argv):
    if argc < 3:
        sys.exit(1)

    device = torch.device('cpu')
    model = torch.jit.load(argv[1]).to(device)
    model.eval()

    info = model.info()

    height = info['height']
    width = info['width']
    fade = bool(info['fade'])

    snake = SnakeMDP(height, width, 0, 0, 0, fade=fade)

    iterations = int(argv[2])
    cb = lambda i, score: print(f'simulated game {i}: {score}')

    scores = simulate(model, device, snake, iterations, ttl=5*width*height, cb=cb)

    print('-' * 100)
    print(f'max score: {max(scores)}')
    print(f'median score: {median(scores)}')
    print(f'min score: {min(scores)}')
    print(f'average score: {sum(scores) / len(scores)}')

    plt.hist(scores, bins=np.linspace(min(scores) - .5, max(scores) + .5, max(scores) - min(scores) + 2))
    plt.show()


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
