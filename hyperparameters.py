from collections import namedtuple
from threading import Lock, Thread
from copy import deepcopy

class Hyperparameters:

    def __init__(self, **kwargs):
        self.state = kwargs
        self.state_lock = Lock()

        self.thread = Thread(target=Hyperparameters.update_routine, args=(self.state, self.state_lock))
        self.thread.start()

    @staticmethod
    def update_routine(state, state_lock):
        while True:
            try:
                cmd = input('\>').split()
                state_lock.acquire()
                if cmd[0] == 'set' and cmd[1] in state.keys():    
                    state[cmd[1]] = cmd[2]
            except:
                pass
            finally:
                print(state)
                state_lock.release()
    
    def __getitem__(self, key):
        self.state_lock.acquire()
        item = self.state[key]
        self.state_lock.release()
        return item
    