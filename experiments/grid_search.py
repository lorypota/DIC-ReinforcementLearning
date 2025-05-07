import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from world import Environment
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.mc_onpolicy import MonteCarloOnPolicyAgent

from experiments.helper import (
    extract_policy_from_vi_agent,
    run_grid_search
)

def main():
    from pathlib import Path

    grid_fp = Path("../grid_configs/A1_grid.npy")
    sigma = 0.1

    env = Environment(grid_fp=grid_fp, no_gui=True, sigma=sigma)

    vi_agent = ValueIterationAgent(str(grid_fp), sigma=sigma, gamma=0.9, theta=1e-6, patience=3)
    state = env.reset(agent_start_pos=env.agent_start_pos)
    for _ in range(1000):
        action = vi_agent.take_action(state)
        state, reward, terminated, info = env.step(action)
        vi_agent.update(state, reward, info.get("actual_action", action))
        if vi_agent.has_converged:
            break
        if terminated:
            state = env.reset(agent_start_pos=env.agent_start_pos)
    optimal_policy = extract_policy_from_vi_agent(vi_agent)

    mc_param_grid = {
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3],
        'alpha':   [0.005, 0.01, 0.05, 0.1, 0.2],
        'gamma':   [0.85, 0.9, 0.95]
    }
    run_grid_search(MonteCarloOnPolicyAgent, env, optimal_policy, mc_param_grid, "Monte Carlo")

    q_param_grid = {
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3],
        'alpha':   [0.005, 0.01, 0.05, 0.1, 0.2],
        'gamma':   [0.85, 0.9, 0.95]
    }
    run_grid_search(QLearningAgent, env, optimal_policy, q_param_grid, "Q-learning")

if __name__ == '__main__':
    main()