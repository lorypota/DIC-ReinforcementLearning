# agents/value_iteration_agent.py
import numpy as np
from .base_agent import BaseAgent
from world.helpers import action_to_direction

class ValueIterationAgent(BaseAgent):
    def __init__(self, grid, gamma=0.9, theta=1e-4):
        super().__init__()
        self.grid = grid
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.actions = [0, 1, 2, 3]  # Down, Up, Left, Right
        self.V = np.zeros(grid.shape)  # Value function
        self.policy = np.zeros(grid.shape, dtype=int)  # Policy
        self._compute_value_function()

    def _compute_value_function(self):
        """Performs Value Iteration to compute the optimal value function and policy."""
        while True:
            delta = 0
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j] in [1, 2, 3]: 
                        continue
                    v = self.V[i, j]
                    q_values = []
                    for a in self.actions:
                        direction = action_to_direction(a)
                        next_pos = (i + direction[0], j + direction[1])
                        # Check if next position is within bounds
                        if (0 <= next_pos[0] < self.grid.shape[0] and 
                            0 <= next_pos[1] < self.grid.shape[1]):
                            # Compute reward based on environment's default reward function
                            if self.grid[next_pos] == 0:
                                reward = -1
                            elif self.grid[next_pos] in [1, 2]:
                                reward = -5
                                next_pos = (i, j)  # Stay in place if hitting obstacle
                            elif self.grid[next_pos] == 3:
                                reward = 10
                            else:
                                continue
                            q = reward + self.gamma * self.V[next_pos]
                        else:
                            q = -5 + self.gamma * self.V[i, j] 
                        q_values.append(q)
                    self.V[i, j] = max(q_values) if q_values else self.V[i, j]
                    self.policy[i, j] = self.actions[np.argmax(q_values)] if q_values else 0
                    delta = max(delta, abs(v - self.V[i, j]))
            if delta < self.theta:
                break

    def take_action(self, state: tuple[int, int]) -> int:
        """Returns the action based on the computed policy."""
        return self.policy[state]

    def update(self, state: tuple[int, int], reward: float, action: int):
        """No update needed as Value Iteration is precomputed."""
        pass