from snakeMDP import Action, SnakeMDP

snake = SnakeMDP(10, 10, 20.0, -100.0, 0.1)

state = snake.sample_start_state()

print(state.world)

while state:
    print(f'{snake.reward(state, Action.LEFT)}')
    state = snake.next(state, Action.LEFT)
    if state:
        print(state.world)
