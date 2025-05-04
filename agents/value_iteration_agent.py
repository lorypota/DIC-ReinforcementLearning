"""Value Iteration Agent.

A planning agent that pre-computes an optimal policy using Value Iteration on the 
known grid of the environment. The agent then executes the policy at run-time.
"""
import numpy as np

from agents import BaseAgent
from world.helpers import action_to_direction

class ValueIterationAgent(BaseAgent):

    # reward function from environment.py
    _REWARD_MAP = {
        0: -1.0,   # empty step cost
        1: -5.0,   # boundary / wall
        2: -5.0,   # obstacle
        3: 10.0,   # target reached
    }

    def __init__(self, grid_cells, gamma: float = 0.95, theta: float = 1e-6):
        """Compute an optimal deterministic policy using Value Iteration.

        Args:
            grid: The grid that describes the world layout
            gamma: Discount factor
            theta: Convergence tolerance
        """
        super().__init__()
        self.gamma = gamma
        self.theta = theta
        self.cells = grid_cells

        self.n_rows, self.n_cols = self.cells.shape
        self.action_space: list[int] = list(range(4))

        self._v = np.zeros_like(self.cells, dtype=float)
        self._policy = np.zeros_like(self.cells, dtype=int)
        self._run_value_iteration()

    def take_action(self, state: tuple[int, int]) -> int:
        return self._policy[state]

    def update(self, state: tuple[int, int], reward: float, action: int):
        pass

    def _run_value_iteration(self):
        """Compute the optimal policy using Value Iteration"""
        # compute the value function
        while True:
            delta = 0.0
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    s = (r, c)
                    if self._is_blocked(s) or self._is_terminal(s):
                        continue

                    v_old = self._v[s]
                    q_values = [self._q_value(s, a) for a in self.action_space]
                    self._v[s] = max(q_values)
                    delta = max(delta, abs(v_old - self._v[s]))
            if delta < self.theta:
                break

        # compute the greedy policy
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = (r, c)
                if self._is_blocked(s) or self._is_terminal(s):
                    continue
                q_values = [self._q_value(s, a) for a in self.action_space]
                self._policy[s] = int(np.argmax(q_values))

    def _q_value(self, state: tuple[int, int], action: int) -> float:
        """One‑step look‑ahead value for (state, action)"""
        dr, dc = action_to_direction(action)
        intended_r = state[0] + dr
        intended_c = state[1] + dc

        # check if intended cell is out of bounds
        out_of_bounds = (
            intended_r < 0 or intended_r >= self.n_rows or
            intended_c < 0 or intended_c >= self.n_cols
        )
        if out_of_bounds:
            reward = self._REWARD_MAP[1] # boundary penalty
            next_state = state
            return reward + self.gamma * self._v[next_state]

        intended_cell_code = self.cells[intended_r, intended_c]
        reward = self._REWARD_MAP[intended_cell_code]

        next_state = state if intended_cell_code in (1, 2) else (intended_r, intended_c)

        return reward + self.gamma * self._v[next_state]

    def _is_blocked(self, state: tuple[int, int]) -> bool:
        return self.cells[state] in (1, 2)
    
    def _is_terminal(self, state: tuple[int, int]) -> bool:
        return self.cells[state] == 3
