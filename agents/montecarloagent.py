"""Monte Carlo Agent implementation."""
from collections import defaultdict
import numpy as np
from .base_agent import BaseAgent

class MonteCarloAgent2(BaseAgent):
    def __init__(self, epsilon=0.1, gamma=0.9):
        super().__init__()
        self.epsilon = epsilon  
        self.gamma = gamma     
        self.Q = defaultdict(float)  
        self.returns = defaultdict(list) 
        self.episode_transitions = [] 
        self.prev_state = None         
        self.initial_state = None     

    def take_action(self, state: tuple[int, int]) -> int:
        
        if self.initial_state is None:
            self.initial_state = state
        elif state == self.initial_state:
            self._process_episode()
            self.initial_state = state
            self.episode_transitions = []

        
        self.prev_state = state

        
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 4) 
        else:
            
            q_values = [self.Q[(state, a)] for a in range(4)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(best_actions)
        return action

    def update(self, state: tuple[int, int], reward: float, action: int):
        
        if self.prev_state is not None:
            self.episode_transitions.append((self.prev_state, action, reward))
        self.prev_state = None 

    def _process_episode(self):
        """Processes the collected episode data to update Q-values."""
        G = 0  
        
        for t in reversed(range(len(self.episode_transitions))):
            state, action, reward = self.episode_transitions[t]
            G = self.gamma * G + reward
            self.returns[(state, action)].append(G)
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])