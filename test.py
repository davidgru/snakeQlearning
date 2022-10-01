import sys
from time import sleep


import torch

from display import Display
from snakeMDP import Action, SnakeMDP
from dqn import ResNet


def test(policy_network, display, snake, delay = 0.5, ttl = 1000):

    state = snake.sample_start_state()
    score = 0

    ttll = ttl

    while state is not None:
        display.draw(state.world, "testing")
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.world).unsqueeze(0).unsqueeze(0)
        action = Action(policy_network(state_tensor).max(1)[1].view(1, 1).item())

        new_score, state = snake.next(state, action)

        if new_score > score:
            ttll = ttl

        ttll -= 1
        if ttll <= 0:
            return score
        sleep(delay)

    return score


def main(argc, argv):
    if argc < 2:
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(argv[1])
    model.eval()

    display = Display(10, 10, 40)

    snake = SnakeMDP(10, 10, 0, 0, 0)

    test(model, display, snake, 0.25)




if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
