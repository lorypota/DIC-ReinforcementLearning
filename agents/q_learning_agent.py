# agents/q_learning_agent.py
import numpy as np
from .base_agent import BaseAgent
import random
from world.helpers import action_to_direction

class QLearningAgent(BaseAgent):
    def __init__(self, grid, gamma=0.9, alpha=0.1, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__()
        self.grid = grid
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = [0, 1, 2, 3]
        self.Q = np.zeros((grid.shape[0], grid.shape[1], len(self.actions)))

    def take_action(self, state: tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.Q[state[0], state[1]])

    def update(self, state: tuple[int, int], reward: float, action: int):
        # Simulate the next state based on the action
        direction = action_to_direction(action)
        next_pos = (state[0] + direction[0], state[1] + direction[1])
        # Check if the next position is valid
        if (0 <= next_pos[0] < self.grid.shape[0] and 
            0 <= next_pos[1] < self.grid.shape[1] and 
            self.grid[next_pos] not in [1, 2]):  # Not a wall or obstacle
            next_state = next_pos
        else:
            next_state = state  # Stay in place if invalid move
        # Q-Learning update
        next_action = np.argmax(self.Q[next_state[0], next_state[1]])
        td_target = reward + self.gamma * self.Q[next_state[0], next_state[1], next_action]
        self.Q[state[0], state[1], action] += self.alpha * (td_target - self.Q[state[0], state[1], action])
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)