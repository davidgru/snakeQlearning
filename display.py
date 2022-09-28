import pygame
import sys

from snakeMDP import BODY_ID, HEAD_ID, FOOD_ID, EMPTY_ID


class Display:

    def __init__(self, height, width, tile_size_px):
        self.height = height
        self.width = width
        self.tile_size = tile_size_px

        surface_height = height * tile_size_px
        surface_width = width * tile_size_px
        
        self.surface = pygame.display.set_mode((surface_width, surface_height))


    def draw(self, world):
        self.surface.fill((0, 0, 0)) # black background
        for y, row in enumerate(world):
            for x, id_ in enumerate(row):
                if id_ == BODY_ID:
                    color = (255, 255, 255) # white
                elif id_ == HEAD_ID:
                    color = (255, 0, 0) # red
                elif id_ == FOOD_ID:
                    color = (0, 255, 0) # green
                else:
                    continue # dont draw empty tiles

                rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                pygame.draw.rect(self.surface, color, rect)
        pygame.display.flip()


    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

