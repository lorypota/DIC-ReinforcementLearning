import matplotlib.pyplot as plt
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
    train_and_evaluate
)  

def main():
    from pathlib import Path

    grid_fp = Path("grid_configs/A1_grid.npy")
    sigma = 0.1
    max_steps = 500
    episodes = 5000
    eval_interval = 100

    env = Environment(grid_fp=grid_fp, no_gui=True, sigma=sigma)

    # Value Iteration to get optimal policy
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

    # Train and evaluate Monte Carlo agent (using hyperparameters from grid search)
    mc_kwargs = dict(gamma=0.9, epsilon=0.3, alpha=0.05, convergence_tol=1e-6, patience=50)
    mc_checkpoints, mc_agreements = train_and_evaluate(
        MonteCarloOnPolicyAgent, mc_kwargs, env, optimal_policy,
        episodes, max_steps, eval_interval
    )

    # Train and evaluate Q-learning agent (using hyperparameters from grid search)
    q_kwargs = dict(gamma=0.9, epsilon=0.05, alpha=0.1, convergence_tol=1e-6, patience=50)
    q_checkpoints, q_agreements = train_and_evaluate(
        QLearningAgent, q_kwargs, env, optimal_policy,
        episodes, max_steps, eval_interval
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(mc_checkpoints, mc_agreements, label="Monte Carlo")
    plt.plot(q_checkpoints, q_agreements, label="Q-learning")
    plt.axhline(1.0, linestyle='--', color='gray', label='Optimal Policy')
    plt.xlabel("Episodes")
    plt.ylabel("Policy Agreement")
    plt.title("Policy Convergence to VI Optimal Policy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()