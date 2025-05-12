import numpy as np
import random
from agents.base_agent import BaseAgent

class MonteCarloAgent(BaseAgent):
    def __init__(self, grid, gamma=0.9, epsilon=0.1, max_episode_length=100):
        super().__init__()
        self.grid = grid
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episode_length = max_episode_length
        self.actions = [0, 1, 2, 3]  
        self.Q = np.zeros((grid.shape[0], grid.shape[1], len(self.actions)))
        self.returns = { (i, j, a): [] 
                         for i in range(grid.shape[0]) 
                         for j in range(grid.shape[1]) 
                         for a in self.actions }
        self.policy = np.random.randint(0, len(self.actions), size=grid.shape)
        self.episode = []  

    def take_action(self, state: tuple[int, int]) -> int:
        """Select action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.policy[state]

    def update(self, state: tuple[int, int], reward: float, action: int):
        """Record state-action-reward for Monte Carlo episode."""
        self.episode.append((state, action, reward))

    def end_episode(self):
        """Performs the First-Visit Monte Carlo update."""
        G = 0
        visited = set()
        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = reward + self.gamma * G
            key = (state[0], state[1], action)
            if key not in visited:
                visited.add(key)
                self.returns[key].append(G)
                self.Q[state[0], state[1], action] = np.mean(self.returns[key])
                self.policy[state] = np.argmax(self.Q[state[0], state[1]])
        self.episode.clear()
