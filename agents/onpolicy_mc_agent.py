import random
import numpy as np
from agents import BaseAgent
from world.grid import Grid

class MonteCarloAgent(BaseAgent):
    """
    On-Policy First-Visit Monte Carlo Control with epsilon-greedy policy.
    """
    def __init__(self, env, gamma: float = 0.9, epsilon: float = 0.1, max_episode_length: int = None):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        # Determine max_episode_length from grid size if not provided
        grid_meta = Grid.load_grid(env.grid_fp)
        cells     = grid_meta.cells
        rows, cols = cells.shape
        self.max_episode_length = max_episode_length or (rows * cols * 2)
        # Load object codes for boundaries to identify valid states
        boundary_code = grid_meta.objects['boundary']
        # Initialize Q(s,a), returns_sum(s,a), returns_count(s,a), and policy
        self.Q = {}            # state tuple -> list of 4 action-values
        self.returns_sum = {}  # state tuple -> list of cumulative returns
        self.returns_count = {}# state tuple -> list of counts
        self.policy = {}       # state tuple -> list of action probabilities
        for r in range(rows):
            for c in range(cols):
                if cells[r,c] != boundary_code:
                    state = (r, c)
                    self.Q[state] = [0.0]*4
                    self.returns_sum[state] = [0.0]*4
                    self.returns_count[state] = [0]*4
                    # start with uniform random policy
                    self.policy[state] = [1/4]*4

    def _generate_episode(self):
        """Runs one episode following current policy."""
        episode = []  # list of (state, action, reward)
        state = self.env.reset()
        for t in range(self.max_episode_length):
            action = self._choose_action(tuple(state))
            next_state, reward, done, info = self.env.step(action)
            episode.append((tuple(state), action, reward))
            state = next_state
            if done:
                break
        return episode

    def _choose_action(self, state):
        # epsilon-greedy selection
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        # choose among those with max Q
        q_vals = self.Q[state]
        max_q = max(q_vals)
        best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
        return random.choice(best_actions)

    def take_action(self, state):
        # used by train.py to interact; we ignore state arg here
        return self._choose_action(tuple(state))

    def update(self, state, reward, action):
        # MC updates only at episode end, so no-op here
        pass

    def learn(self, num_episodes: int):
        """Perform num_episodes of first-visit MC updates."""
        for _ in range(num_episodes):
            episode = self._generate_episode()
            G = 0.0
            visited = set()
            # Traverse backwards
            for i in reversed(range(len(episode))):
                s, a, r = episode[i]
                G = self.gamma * G + r
                # first-visit check
                if (s, a) not in visited:
                    visited.add((s, a))
                    idx = a  # action index
                    self.returns_sum[s][idx] += G
                    self.returns_count[s][idx] += 1
                    # update Q
                    self.Q[s][idx] = self.returns_sum[s][idx] / self.returns_count[s][idx]
                    # update policy to be epsilon-greedy
                    q_vals = self.Q[s]
                    best_idx = int(np.argmax(q_vals))
                    for ai in range(4):
                        if ai == best_idx:
                            self.policy[s][ai] = 1 - self.epsilon + (self.epsilon/4)
                        else:
                            self.policy[s][ai] = self.epsilon/4

    def get_policy(self):
        """Return the greedy policy (action with highest Q) for each state."""
        greedy = {}
        for s, probs in self.policy.items():
            greedy[s] = int(max(range(4), key=lambda a: self.Q[s][a]))
        return greedy
