import numpy as np
from collections import deque, namedtuple
from enum import IntEnum

EMPTY_ID = -0.1
BODY_ID = 1
HEAD_ID = 2
FOOD_ID = 3

State = namedtuple('State', 'world head_y head_x food_y food_x body')
Action = IntEnum('Action', 'LEFT RIGHT DOWN UP')

class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3



def get_action(n):
    if n == 0:
        return Action.LEFT

class SnakeMDP:

    def __init__(self, height, width, food_reward, death_reward, time_reward, fade=False):
        self.height = height
        self.width = width
        self.food_reward = food_reward
        self.death_reward = death_reward
        self.time_penalty = time_reward
        self.fade = fade


    def sample_start_state(self):
        world = np.full(shape=(self.height, self.width), fill_value=EMPTY_ID, dtype=np.float32)
        start_y = np.random.randint(self.height)
        start_x = np.random.randint(self.width)
        world[start_y, start_x] = HEAD_ID
        food_y, food_x = self._place_food(world)
        body = deque([(start_y, start_x)])

        return State(world, start_y, start_x, food_y, food_x, body)


    def next(self, state, action):
        """Find the next state given state and action"""

        world, head_y, head_x, food_y, food_x, body = state
        new_head_y, new_head_x = self.find_next_head_pos(head_y, head_x, action)

        if self._crashed(world, new_head_y, new_head_x, body):
            return len(state.body) - 1, None # Game over

        new_world = world.copy()
        new_body = body.copy()

        new_world = np.full(shape=(self.height, self.width), fill_value=EMPTY_ID, dtype=np.float32)
        new_body = body.copy()
        new_body.appendleft((new_head_y, new_head_x))

        found_food = self._found_food(world, new_head_y, new_head_x)
        if not found_food:
            new_body.pop()

        # redraw world
        for i, pos in enumerate(new_body):
            y, x = pos
            if self.fade:
                new_world[y, x] = float(BODY_ID * (len(new_body) - i)) / len(new_body)
            else:
                new_world[y, x] = BODY_ID
        new_world[new_head_y, new_head_x] = HEAD_ID

        if found_food:
            food_pos = self._place_food(new_world)
            if not food_pos:
                return len(new_body) - 1, None # Won
            food_y, food_x = food_pos

        new_world[food_y, food_x] = FOOD_ID

        return len(new_body) - 1, State(new_world, new_head_y, new_head_x, food_y, food_x, new_body)


    def reward(self, state, action):
        """Find the reward given state and action"""
        
        world, head_y, head_x, _, _, body = state
        new_head_y, new_head_x = self.find_next_head_pos(head_y, head_x, action)
        if self._crashed(world, new_head_y, new_head_x, body):
            return self.death_reward
        if self._found_food(world, new_head_y, new_head_x):
            return self.food_reward
        return self.time_penalty


    def find_next_head_pos(self, head_y, head_x, action):
        """Compute the next state"""

        if action == Action.LEFT:
            head_x -= 1
        elif action == Action.RIGHT:
            head_x  += 1
        elif action == Action.DOWN:
            head_y += 1
        elif action == Action.UP:
            head_y -= 1
        else:
            raise "Invalid Action!"
        return (head_y, head_x)


    def _crashed(self, world, head_y, head_x, body):
        if not (0 <= head_y < self.height and 0 <= head_x < self.width):
            return True
        return 0.0 <= world[head_y,head_x] <= BODY_ID


    def _found_food(self, world, head_y, head_x):
        return np.isclose(world[head_y,head_x], FOOD_ID, atol=0.1)


    def _place_food(self, world):
        """Find a free spot for the food"""

        num_empty = (world < 0).sum()
        if num_empty <= 0:
            return None
        food_index = np.random.randint(num_empty)
        zero_count = 0
        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                if world[y, x] < 0:
                    if zero_count == food_index:
                        world[y, x] = FOOD_ID
                        return (y, x)
                    zero_count += 1
        return None
