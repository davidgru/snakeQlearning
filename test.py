
import torch

from snakeMDP import Action

def test(policy_network, display, snake, ttl = 1000):

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

    return score