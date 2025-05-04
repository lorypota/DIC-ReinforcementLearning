from agents.base_agent import BaseAgent
from world.grid import Grid
from world.environment import Environment
from world.helpers import action_to_direction
import numpy as np

class ValueIterationAgent(BaseAgent):
    """
    Implements the Value Iteration algorithm for a given grid MDP.

    Args:
        grid_fp (str): Path to the numpy grid configuration file.
        sigma (float): Probability of taking a random action instead of intended. Default 0.0.
        gamma (float): Discount factor in [0,1). Default 0.9.
        theta (float): Convergence threshold for max change in V. Default 1e-6.
        patience (int): Number of consecutive sweeps below theta to declare convergence. Default 1.
    """
    def __init__(self, grid_fp: str, sigma: float = 0.0,
                 gamma: float = 0.9, theta: float = 1e-6, patience: int = 1):
        super().__init__()
        grid_obj = Grid.load_grid(grid_fp)
        self.grid = grid_obj.cells
        self.n_cols, self.n_rows = self.grid.shape
        self.sigma = sigma
        self.gamma = gamma
        self.theta = theta
        self.patience = patience
        self.actions = [0, 1, 2, 3]
        self.reward_fn = Environment._default_reward_function
        self.states = []
        self.terminal_states = set()
        for x in range(self.n_cols):
            for y in range(self.n_rows):
                v = self.grid[x, y]
                if v not in (1, 2):
                    self.states.append((x, y))
                    if v == 3:
                        self.terminal_states.add((x, y))
        self.V = {s: 0.0 for s in self.states}
        self.policy = {s: 0 for s in self.states}
        self._unchanged_count = 0
        self.has_converged = False
        self.P = {}
        for s in self.states:
            for a in self.actions:
                self.P[(s, a)] = self._compute_transitions(s, a)

    def _compute_transitions(self, state, action):
        """
        Compute transition probabilities, next states, and rewards for a given state and action.

        Args:
            state (tuple): Current grid coordinates (x, y).
            action (int): Intended action index.

        Returns:
            List of tuples (probability, next_state, reward).
        """
        transitions = []
        p_intended = 1 - self.sigma + (self.sigma / 4)
        p_other = self.sigma / 4
        for a2 in self.actions:
            prob = p_intended if a2 == action else p_other
            dx, dy = action_to_direction(a2)
            nxt = (state[0] + dx, state[1] + dy)
            if (nxt[0] < 0 or nxt[0] >= self.n_cols or
                nxt[1] < 0 or nxt[1] >= self.n_rows or
                self.grid[nxt] in (1, 2)):
                nxt_state = state
            else:
                nxt_state = nxt
            r = self.reward_fn(self.grid, nxt_state)
            transitions.append((prob, nxt_state, r))
        return transitions

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Select the greedy action according to the current policy.

        Args:
            state (tuple): Current grid coordinates (x, y).

        Returns:
            int: Selected action index.
        """
        return self.policy.get(tuple(state), 0)

    def update(self, state: tuple[int, int], reward: float, action: int):
        """
        Perform one sweep of Bellman backups over all states and update policy.
        Check convergence and mark the agent as converged if criteria met.

        Args:
            state (tuple): Last observed state (unused in VI step-by-sweep).
            reward (float): Last received reward (unused in VI planning).
            action (int): Last taken action (unused in VI planning).
        """

        delta = 0.0
        for s in self.states:
            if s in self.terminal_states:
                continue
            v_old = self.V[s]
            q_vals = [
                sum(p * (r + self.gamma * self.V[s2]) for (p, s2, r) in self.P[(s, a)])
                for a in self.actions
            ]
            self.V[s] = max(q_vals)
            delta = max(delta, abs(v_old - self.V[s]))

        for s in self.states:
            if s in self.terminal_states:
                continue
            q_vals = [
                sum(p * (r + self.gamma * self.V[s2]) for (p, s2, r) in self.P[(s, a)])
                for a in self.actions
            ]
            self.policy[s] = int(np.argmax(q_vals))

        if delta < self.theta:
            self._unchanged_count += 1
        else:
            self._unchanged_count = 0
        if self._unchanged_count >= self.patience:
            self.has_converged = True
            print(f"VI Agent converged (tol={self.theta}, patience={self.patience})")