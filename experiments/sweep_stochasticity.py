from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from world.environment import Environment

from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.mc_onpolicy import MonteCarloOnPolicyAgent
from experiments.helper import extract_policy_from_vi_agent, train_and_evaluate

def run_vi_and_get_policy(grid_fp, sigma):
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
    return env, extract_policy_from_vi_agent(vi_agent)

def sweep_sigma(grid_fp):
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    q_agreements_final = []
    mc_agreements_final = []

    for sigma in sigmas:
        print(f"\n=== Running for sigma = {sigma} ===")
        env, optimal_policy = run_vi_and_get_policy(grid_fp, sigma)

        # Q-learning

        #plugging best values on the basis of grid search 
        q_params = dict(gamma=0.85, epsilon=0.05, alpha=0.2, convergence_tol=1e-6, patience=50)
        _, q_agreements = train_and_evaluate(QLearningAgent, q_params, env, optimal_policy, 5000, 500, 100)
        q_agreements_final.append(q_agreements[-1] if q_agreements else 0)

        # Monte Carlo
        mc_params = dict(gamma=0.9, epsilon=0.01, alpha=0.1, convergence_tol=1e-6, patience=50)
        _, mc_agreements = train_and_evaluate(MonteCarloOnPolicyAgent, mc_params, env, optimal_policy, 5000, 500, 100)
        mc_agreements_final.append(mc_agreements[-1] if mc_agreements else 0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, q_agreements_final, marker='o', label="Q-learning")
    plt.plot(sigmas, mc_agreements_final, marker='s', label="Monte Carlo")
    plt.axhline(1.0, linestyle='--', color='gray', label='Optimal Policy')
    plt.xlabel("Sigma (Environment Stochasticity)")
    plt.ylabel("Final Policy Agreement")
    plt.title("Agent Robustness to Stochasticity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid_fp = Path("../grid_configs/A1_grid.npy")  
    sweep_sigma(grid_fp)
