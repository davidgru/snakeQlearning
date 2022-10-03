import matplotlib.pyplot as plt
from statistics import median
import sys
import torch

from dqn import CNN
from snakeMDP import SnakeMDP
from test import test

# simulates specified number of games and returns a list of scores
def simulate(policy_network, snake, iterations=0, cb=None):
    scores = []
    for i in range(iterations):
        stats = test(policy_network, snake)
        scores.append(stats.score)
        if cb:
            cb(i, stats.score)
    return scores


def main(argc, argv):
    if argc < 3:
        sys.exit(1)

    device = torch.device('cpu')
    policy_network = CNN(10, 10, device).to(device)
    policy_network.load_state_dict(torch.load(argv[1], map_location=device))
    policy_network.eval()

    snake = SnakeMDP(10, 10, 0, 0, 0)
    iterations = int(argv[2])
    cb = lambda i, score: print(f'simulated game {i}: {score}')

    scores = simulate(policy_network, snake, iterations, cb)

    print('-' * 100)
    print(f'max score: {max(scores)}')
    print(f'median score: {median(scores)}')
    print(f'min score: {min(scores)}')
    print(f'average score: {sum(scores) / len(scores)}')

    plt.hist(scores, bins=max(scores) - min(scores) + 1)
    plt.show()


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
