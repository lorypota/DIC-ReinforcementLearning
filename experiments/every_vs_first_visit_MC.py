from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from world.environment import Environment

from agents.value_iteration import ValueIterationAgent
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

def compare_first_vs_every_visit(grid_fp, episodes=5000, max_steps=500, eval_interval=100):
    sigma = 0.1
    env, optimal_policy = run_vi_and_get_policy(grid_fp, sigma)

    common_args = dict(
        gamma=0.9,
        epsilon=0.1,
        alpha=0.1,
        max_episode_len=max_steps,
        convergence_tol=1e-6,
        patience=50
    )

    # First-visit
    fv_checkpoints, fv_agreements = train_and_evaluate(
        MonteCarloOnPolicyAgent,
        {**common_args, "first_visit": True},
        env,
        optimal_policy,
        episodes=episodes,
        max_steps=max_steps,
        eval_interval=eval_interval
    )

    # Every-visit
    ev_checkpoints, ev_agreements = train_and_evaluate(
        MonteCarloOnPolicyAgent,
        {**common_args, "first_visit": False},
        env,
        optimal_policy,
        episodes=episodes,
        max_steps=max_steps,
        eval_interval=eval_interval
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ev_checkpoints, ev_agreements, label="Every-Visit MC", marker='o')
    plt.plot(fv_checkpoints, fv_agreements, label="First-Visit MC", marker='s')
    plt.axhline(1.0, linestyle='--', color='gray', label='Optimal Policy')
    plt.xlabel("Episodes")
    plt.ylabel("Policy Agreement")
    plt.title("First-Visit vs Every-Visit Monte Carlo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid_fp = Path("../grid_configs/large_grid.npy")
    compare_first_vs_every_visit(grid_fp)
