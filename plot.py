
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class GameStats:

    def __init__(self):
        self.duration = 0
        self.score = 0
        self.cum_score = 0

    def push(self, score):
        self.duration += 1
        self.score = score
        self.cum_score += score


class MetricHistory:

    def __init__(self, name, granularity):
        self.name = name
        self.granularity = granularity
        self.data = deque([], maxlen=granularity)
        self.avg = []
        self.max = []
        self.min = []

    def push(self, x):
        self.data.append(x)
        if len(self.data) % self.granularity == 0:
            self.max.append(max(self.data))
            self.avg.append(sum(self.data) / self.granularity)
            self.min.append(min(self.data))
            self.data.clear()


class Plot:

    def __init__(self, granularity):
        self.max_score = []
        self.avg_score = []
        self.min_score = []

        self.granularity = granularity

        metric_names = ['score', 'cumulative_score']

        self.metrics = {}


        fig, axs = plt.subplots(len(metric_names))

        for i, name in enumerate(metric_names):
            hist = MetricHistory(name, granularity)
            axs[i].set_title(name)
            axs[i].plot([], label='Max', color='g')
            axs[i].plot([], label='Average', color='b')
            axs[i].plot([], label='Min', color='r') 
            axs[i].legend()
            axs[i].grid()
            self.metrics[name] = {
                "hist": hist,
                "ax": axs[i],
            }
        
        plt.ion()
        plt.show()
        plt.pause(.1)
        self.points = 0


    def push(self, game_stats):
        self.metrics['score']['hist'].push(game_stats.score)
        self.metrics['cumulative_score']['hist'].push(game_stats.cum_score)

        self.points += 1
        if self.points % self.granularity == 0:
            for val in self.metrics.values():
                hist = val['hist']
                ax = val['ax']
                ax.cla()
                ax.set_title(hist.name)
                ax.grid()
                ax.plot(hist.max, label='Max', color='g')
                ax.plot(hist.avg, label='Average', color='b')
                ax.plot(hist.min, label='Min', color='r')
            plt.pause(.01)

