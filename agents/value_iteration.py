from agents.base_agent import BaseAgent
import numpy as np

class ValueIterationAgent(BaseAgent):
    def __init__(self, env, gamma=0.9, theta=1e-4):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.grid.shape)
        self.policy = np.zeros(env.grid.shape, dtype=int)
        self.actions = [0, 1, 2, 3]  # Up, Down, Left, Right, adjust as needed
        self.run_value_iteration()

    def run_value_iteration(self):
        while True:
            delta = 0
            for state in self.env.get_all_states():
                if self.env.is_terminal(state):
                    continue
                v = self.V[state]
                q_values = []
                for a in self.actions:
                    transitions = self.env.get_transitions(state, a)
                    q = sum(p * (r + self.gamma * self.V[s_]) for p, s_, r in transitions)
                    q_values.append(q)
                self.V[state] = max(q_values)
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break
        self.extract_policy()

    def extract_policy(self):
        for state in self.env.get_all_states():
            if self.env.is_terminal(state):
                continue
            q_values = []
            for a in self.actions:
                transitions = self.env.get_transitions(state, a)
                q = sum(p * (r + self.gamma * self.V[s_]) for p, s_, r in transitions)
                q_values.append(q)
            self.policy[state] = np.argmax(q_values)

    def take_action(self, state):
        return self.policy[state]

    def update(self, state, reward, action):
        pass  # not needed for Value Iteration (offline planning)
