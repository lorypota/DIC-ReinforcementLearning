from agents import BaseAgent
import numpy as np
from world.grid import Grid
from world.helpers import action_to_direction

class ValueIterationAgent(BaseAgent):
    """
    Value Iteration agent that computes optimal state-values and policy
    given a fully-known environment model (grid, stochasticity, reward_fn).
    """
    def __init__(self, env, gamma: float = 0.9, theta: float = 1e-8):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.theta = theta
        # load metadata for object codes
        grid_meta = Grid.load_grid(env.grid_fp)
        self.boundary_code = grid_meta.objects['boundary']
        self.obstacle_code = grid_meta.objects['obstacle']
        self.target_code   = grid_meta.objects['target']
        self.charger_code  = grid_meta.objects['charger']
        # actual grid after reset is env.grid, but for planning we use initial cells
        self.cells = grid_meta.cells
        # initialize
        self.V = {tuple(s): 0.0 for s in np.argwhere(self.cells != self.boundary_code)}
        self.policy = {}
        self._run_value_iteration()

    def _run_value_iteration(self):
        """Perform the value iteration loop until convergence."""
        states = list(self.V.keys())
        while True:
            delta = 0.0
            for s in states:
                # skip terminal (target) states
                if self.cells[s] == self.target_code:
                    continue
                v_old = self.V[s]
                # evaluate all actions
                action_values = []
                for a in range(4):
                    total = 0.0
                    # stochastic outcomes: actual_action = a with prob 1-sigma, else uniform
                    for ap in range(4):
                        prob = (1 - self.env.sigma) if ap == a else (self.env.sigma / 3)
                        # compute next state for actual_action ap
                        direction = action_to_direction(ap)
                        sp = (s[0] + direction[0], s[1] + direction[1])
                        # check boundaries
                        if (sp[0] < 0 or sp[0] >= self.cells.shape[0] or
                            sp[1] < 0 or sp[1] >= self.cells.shape[1] or
                            self.cells[sp] in (self.boundary_code, self.obstacle_code)):
                            # stay in place on invalid move
                            sp = s
                        # reward for moving into sp
                        r = self.env.reward_fn(self.cells, sp)
                        total += prob * (r + self.gamma * self.V[sp])
                    action_values.append(total)
                # best action value
                v_new = max(action_values)
                best_a = int(np.argmax(action_values))
                self.V[s] = v_new
                self.policy[s] = best_a
                delta = max(delta, abs(v_old - v_new))
            if delta < self.theta:
                break

    def take_action(self, state):
        # return the precomputed optimal action for this state
        return self.policy.get(tuple(state), 0)

    def update(self, state, reward, action):
        # no-op for planning agent
        pass