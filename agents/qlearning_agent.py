import random
import numpy as np
from agents import BaseAgent
from world.grid import Grid

class QLearningAgent(BaseAgent):
    """
    Q-Learning agent with ε-greedy exploration.
    """
    def __init__(self, env, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0.1):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        # Load grid metadata to identify boundary/obstacle states
        grid_meta = Grid.load_grid(env.grid_fp)
        boundary_code = grid_meta.objects['boundary']
        rows, cols = env.grid.shape
        # Initialize Q(s,a) to zero for all non-boundary states
        self.Q = {}
        for r in range(rows):
            for c in range(cols):
                if env.grid[r, c] != boundary_code:
                    self.Q[(r, c)] = [0.0] * 4
        # placeholders to store last transition
        self.last_state = None

    def _choose_action(self, state):
        # ε-greedy selection
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        q_vals = self.Q[state]
        max_q = max(q_vals)
        best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
        return random.choice(best_actions)

    def take_action(self, state):
        # store state for update
        self.last_state = tuple(state)
        action = self._choose_action(self.last_state)
        return action

    def update(self, state, reward, actual_action):
        # only update if we have a stored last state
        if self.last_state is None:
            return
        s0 = self.last_state
        a0 = actual_action  # actual executed action
        s1 = tuple(state)
        # Calculate the TD target using max_a' Q(s1,a')
        q_next = max(self.Q[s1]) if s1 in self.Q else 0.0
        td_target = reward + self.gamma * q_next
        td_error = td_target - self.Q[s0][a0]
        # Update Q-value
        self.Q[s0][a0] += self.alpha * td_error
        # Next update will use the new state
        self.last_state = s1

    def get_policy(self):
        """Return greedy policy (action with highest Q) for each state."""
        policy = {}
        for s, q_vals in self.Q.items():
            best_a = int(np.argmax(q_vals))
            policy[s] = best_a
        return policy
