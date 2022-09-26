from snakeMDP import Action, SnakeMDP
from display import Display

from time import sleep

HEIGHT = 10
WIDTH = 10
TILE_SIZE = 20

snake = SnakeMDP(HEIGHT, WIDTH, 20.0, -100.0, 0.1)
display = Display(HEIGHT, WIDTH, TILE_SIZE)


state = snake.sample_start_state()

while True:
    display.draw(state.world)
    state = snake.next(state, Action.LEFT)

    if not state:
        state = snake.sample_start_state()

    display.update()
    sleep(0.5)
    