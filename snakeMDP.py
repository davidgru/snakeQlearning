import numpy as np
from collections import deque, namedtuple
from enum import IntEnum

EMPTY_ID = -0.1
BODY_ID = 1
HEAD_ID = 2
FOOD_ID = 3

State = namedtuple('State', 'world head_y head_x body')
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

    def __init__(self, height, width, food_reward, death_reward, time_reward):
        self.height = height
        self.width = width
        self.food_reward = food_reward
        self.death_reward = death_reward
        self.time_penalty = time_reward


    def sample_start_state(self):
        world = np.full(shape=(self.height, self.width), fill_value=EMPTY_ID, dtype=np.float32)
        start_y = np.random.randint(self.height)
        start_x = np.random.randint(self.width)
        world[start_y, start_x] = HEAD_ID
        world = self._place_food(world)
        body = deque([(start_y, start_x)])
        return State(world, start_y, start_x, body)


    def next(self, state, action):
        """Find the next state given state and action"""

        world, head_y, head_x, body = state
        new_head_y, new_head_x = self.find_next_head_pos(head_y, head_x, action)

        if self._crashed(world, new_head_y, new_head_x, body):
            return None # Game over

        new_world = world.copy()
        new_body = body.copy()
        
        # advance the snake
        new_body.appendleft((new_head_y, new_head_x))
        new_world[new_head_y, new_head_x] = HEAD_ID
        new_world[head_y, head_x] = BODY_ID

        if world[new_head_y, new_head_x] == FOOD_ID:
            self._place_food(new_world)
        else:
            back_y, back_x = new_body.pop()
            new_world[back_y, back_x] = EMPTY_ID

        return State(new_world, new_head_y, new_head_x, new_body)


    def reward(self, state, action):
        """Find the reward given state and action"""
        
        world, head_y, head_x, body = state
        new_head_y, new_head_x = self.find_next_head_pos(head_y, head_x, action)
        if self._crashed(world, new_head_y, new_head_x, body):
            return self.death_reward
        if world[new_head_y,new_head_x] == FOOD_ID:
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
        return body and world[head_y,head_x] == BODY_ID and (head_y, head_x) != body[-1]


    def _place_food(self, world):
        """Find a free spot for the food"""

        num_empty = (world < 0).sum()
        food_index = np.random.randint(num_empty)
        zero_count = 0
        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                if world[y, x] < 0:
                    if zero_count == food_index:
                        world[y, x] = FOOD_ID
                        return world
                    zero_count += 1
        return world
