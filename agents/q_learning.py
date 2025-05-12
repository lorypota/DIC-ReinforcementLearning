from agents.base_agent import BaseAgent
import numpy as np
import random

class QLearningAgent(BaseAgent):
    def __init__(self, actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Dictionary for Q-values: state -> [action values]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.last_state = None
        self.last_action = None

    def _get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions)
        return self.q_table[state]

    def take_action(self, state):
        q_values = self._get_q_values(state)
        if random.random() < self.epsilon:
            action = random.randint(0, self.actions - 1)
        else:
            action = int(np.argmax(q_values))
        self.last_state = state
        self.last_action = action
        return action

    def update(self, state, reward, action):
        q_values = self._get_q_values(self.last_state)
        next_q_values = self._get_q_values(state)
        td_target = reward + self.gamma * np.max(next_q_values)
        td_error = td_target - q_values[self.last_action]
        q_values[self.last_action] += self.alpha * td_error

