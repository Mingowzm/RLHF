import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

from reward import basic_reward

class MovingDotEnv(gym.Env):
    def __init__(self):
        super(MovingDotEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.array([0, 0])
        return self.state

    def step(self, action):
        if action == 0:  # Up
            self.state[1] = min(self.state[1] + 1, 10)
        elif action == 1:  # Down
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 2:  # Left
            self.state[0] = max(self.state[0] - 1, 0)
        elif action == 3:  # Right
            self.state[0] = min(self.state[0] + 1, 10)
        else:
            raise Exception("Invalid Action")

        reward = basic_reward(self.state)
        # print(f"The reward is {reward}.")
        done = False

        if np.array_equal(self.state, np.array([5.0, 5.0])):
            done = True

        info = {}
        return self.state, reward, done, info

    def render(self, ax):
        ax.scatter(self.state[0], self.state[1], c='red')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)


