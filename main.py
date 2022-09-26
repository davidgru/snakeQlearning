from time import sleep
import torch
import torch.optim as optim

from display import Display
from dqn import DQN
from exploration_strategy import EpsilonGreedyPolicy
from snakeMDP import Action, SnakeMDP

HEIGHT = 10
WIDTH = 10
TILE_SIZE = 20

EXPLORATION_RATE = 0.1

# use qpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device.type)

# create snake mdp
snake = SnakeMDP(HEIGHT, WIDTH, 20.0, -100.0, 0.1)

# create pygame display
display = Display(HEIGHT, WIDTH, TILE_SIZE)

# create policy and target networks
policy_network = DQN(HEIGHT, WIDTH, device).to(device)
target_network = DQN(HEIGHT, WIDTH, device).to(device)
target_network.load_state_dict(policy_network.state_dict())
target_network.eval()

optimizer = optim.RMSprop(policy_network.parameters())

exploration_strategy = EpsilonGreedyPolicy(0.1)

state = snake.sample_start_state()

while True:
    policy_input = torch.from_numpy(state.world)
    policy_input = policy_input.unsqueeze(0)
    policy_input = policy_input.unsqueeze(0)

    action = exploration_strategy.sample_action(policy_network, policy_input)

    print(action)

    display.draw(state.world)
    state = snake.next(state, action)

    if not state:
        state = snake.sample_start_state()

    display.update()
    sleep(0.5)
    