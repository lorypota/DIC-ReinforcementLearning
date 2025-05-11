# agents/monte_carlo_agent.py
import numpy as np
from .base_agent import BaseAgent
import random

class MonteCarloAgent(BaseAgent):
    def __init__(self, grid, gamma=0.9, epsilon=0.1, max_episode_length=100):
        super().__init__()
        self.grid = grid
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episode_length = max_episode_length
        self.actions = [0, 1, 2, 3]
        self.Q = np.zeros((grid.shape[0], grid.shape[1], len(self.actions)))  # Q-values
        self.returns = {(i, j, a): [] for i in range(grid.shape[0]) 
                        for j in range(grid.shape[1]) for a in self.actions}  # Returns for each state-action
        self.policy = np.random.randint(0, len(self.actions), grid.shape)  # Random initial policy

    def take_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.policy[state]

    def update(self, state: tuple[int, int], reward: float, action: int):
        """Store the experience for Monte-Carlo update at the end of the episode."""
        if not hasattr(self, 'episode'):
            self.episode = []
        self.episode.append((state, action, reward))

    def end_episode(self):
        """Perform Monte-Carlo update at the end of an episode."""
        G = 0  # Return
        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward
            self.returns[(state[0], state[1], action)].append(G)
            # First-visit MC: Only update if this is the first occurrence of (state, action)
            self.Q[state[0], state[1], action] = np.mean(self.returns[(state[0], state[1], action)])
            # Update policy to be greedy w.r.t Q
            self.policy[state] = np.argmax(self.Q[state[0], state[1]])
        self.episode = []  # Reset episode