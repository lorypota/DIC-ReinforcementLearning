import numpy as np
from agents import BaseAgent
import random

class MonteCarloOnPolicyAgent(BaseAgent):
    def __init__(self,
                 grid_fp,
                 gamma=0.9,
                 epsilon=0.1,
                 alpha=0.1,
                 max_episode_len=500,
                 convergence_tol=1e-3,
                 patience=5,
                 first_visit=False):
        """
        Initializes the MonteCarloOnPolicyAgent with the given parameters.

        Args:
            grid_fp (str): File path to the NumPy grid configuration.
            gamma (float): Discount factor for future rewards. Defaults to 0.9.
            epsilon (float): Exploration rate for ε-greedy policy. Defaults to 0.1.
            alpha (float): Learning rate for Q-value updates. Defaults to 0.1.
            max_episode_len (int): Maximum length of an episode before forced termination. Defaults to 500.
            convergence_tol (float): Tolerance for convergence check based on Q-values. Defaults to 1e-3.
            patience (int): Number of consecutive unchanged Q-value updates to consider convergence. Defaults to 5.
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.alpha = alpha
        self.max_episode_len = max_episode_len
        self.convergence_tol = convergence_tol
        self.patience = patience
        self.first_visit = first_visit

        self.grid = np.load(grid_fp)
        self.n_rows, self.n_cols = self.grid.shape
        self.actions = [0, 1, 2, 3]
        self.Q = np.zeros((self.n_rows, self.n_cols, len(self.actions)))
        self.episode = []          # stores (state, action, reward)
        self._prev_Q = np.copy(self.Q)
        self._unchanged_count = 0
        self.has_converged = False

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Selects an action using ε-greedy policy based on current Q-values.

        Args:
            state (tuple): The current state position as (row, column).

        Returns:
            int: The chosen action (0: right, 1: left, 2: up, 3: down).
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            i, j = state
            return int(np.argmax(self.Q[i, j]))

    def record(self, state: tuple[int, int], action: int, reward: float):
        """
        Records a transition without updating Q-values until the episode ends.

        Args:
            state (tuple): The state visited.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
        """
        self.episode.append((state, action, reward))

    def end_episode(self):
        """
        Finalizes the current episode by applying the Monte Carlo update
        to all stored transitions and checking for convergence.
        """
        # Decay epsilon after each episode
        #self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self._update_Q()
        self._check_convergence()
        self.episode.clear()

    def update(self, prev_state: tuple[int, int],
                     action: int,
                     reward: float,
                     done: bool):
        """
        Updates the agent with a new experience from the environment.
        If the episode has ended or reached its max length, applies the MC update.

        Args:
            prev_state (tuple): The previous state before taking the action.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode has ended.
        """
        self.record(prev_state, action, reward)

        if done or len(self.episode) >= self.max_episode_len:
            self.end_episode()

    def _update_Q(self):
        """
        Performs the Monte Carlo update for all transitions stored in the episode.
        Updates the Q-values incrementally using the learning rate α.
        """
        G = 0
        visited = set()
        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = reward + self.gamma * G

            if self.first_visit:
                if (state, action) in visited:
                    continue
                visited.add((state, action))

            i, j = state
            old = self.Q[i, j, action]
            self.Q[i, j, action] += self.alpha * (G - old)


    def _check_convergence(self):
        """
        Checks whether Q-values have converged by comparing with previous values.
        If the maximum difference is below the tolerance for a number of
        consecutive episodes (defined by patience), marks the agent as converged.
        """
        diff = np.max(np.abs(self.Q - self._prev_Q))
        if diff < self.convergence_tol:
            self._unchanged_count += 1
        else:
            self._unchanged_count = 0

        self._prev_Q[:] = self.Q
        if self._unchanged_count >= self.patience:
            self.has_converged = True
            print(f"MC Agent converged (tol={self.convergence_tol}, patience={self.patience})")
