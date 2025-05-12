from agents.base_agent import BaseAgent
from world.helpers import action_to_direction
import numpy as np

class ValueIterationAgent(BaseAgent):
    def __init__(self, env, gamma=0.9, theta=1e-4):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.grid.shape)
        self.policy = np.zeros(env.grid.shape, dtype=int)
        self.actions = [0, 1, 2, 3]
        self.run_value_iteration()

    def run_value_iteration(self):
        while True:
            delta = 0
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j] in [1, 2, 3]:
                        continue

                    old_value = self.V[i, j]
                    q_values = []

                    for a_idx, a in enumerate(self.actions):
                        direction = action_to_direction(a)
                        ni, nj = i + direction[0], j + direction[1]

                        if 0 <= ni < self.grid.shape[0] and 0 <= nj < self.grid.shape[1]:
                            cell = self.grid[ni, nj]
                            if cell == 0:
                                reward = -1
                                next_pos = (ni, nj)
                            elif cell in [1, 2]:
                                reward = -5
                                next_pos = (i, j)
                            elif cell == 3:
                                reward = 10
                                next_pos = (ni, nj)
                            else:
                                continue
                        else:
                            reward = -5
                            next_pos = (i, j)

                        q = reward + self.gamma * self.V[next_pos]
                        q_values.append(q)

                    if q_values:
                        self.V[i, j] = max(q_values)
                        self.policy[i, j] = self.actions[np.argmax(q_values)]
                    else:
                        self.V[i, j] = old_value
                        self.policy[i, j] = 0

                    delta = max(delta, abs(old_value - self.V[i, j]))

            if delta < self.theta:
                break

    def take_action(self, state):
        return self.policy[state]

    def update(self, state, reward, action):
        pass  # not needed for Value Iteration (offline planning)
