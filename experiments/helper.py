import numpy as np
from tqdm import trange
from itertools import product
import matplotlib.pyplot as plt

from agents.mc_onpolicy import MonteCarloOnPolicyAgent

def compute_policy_agreement(q_values, optimal_policy):
    greedy_policy = np.argmax(q_values, axis=-1)
    return np.mean(greedy_policy == optimal_policy)

def extract_policy_from_vi_agent(agent):
    grid = agent.grid
    n_rows, n_cols = grid.shape
    policy = np.zeros((n_rows, n_cols), dtype=int)
    for (x, y), action in agent.policy.items():
        policy[x, y] = action
    return policy

def train_and_evaluate(agent_class, agent_kwargs, env, optimal_policy, episodes, max_steps, eval_interval):
    agent = agent_class(str(env.grid_fp), **agent_kwargs)
    agreements = []
    checkpoints = []

    for episode in trange(episodes, desc=f"Training {agent_class.__name__}"):
        state = env.reset(agent_start_pos=env.agent_start_pos)
        for _ in range(max_steps):
            action = agent.take_action(state)
            next_state, reward, terminated, info = env.step(action)
            actual_action = info.get("actual_action", action)

            if isinstance(agent, MonteCarloOnPolicyAgent):
                agent.update(
                    prev_state=state,
                    action=actual_action,
                    reward=reward,
                    done=terminated
                )
            else:
                agent.update(
                    prev_state=state,
                    action=actual_action,
                    reward=reward,
                    done=terminated,
                    next_state=next_state
                )

            state = next_state
            if terminated or getattr(agent, 'has_converged', False):
                break

        if episode % eval_interval == 0 or getattr(agent, 'has_converged', False):
            agreement = compute_policy_agreement(agent.Q, optimal_policy)
            agreements.append(agreement)
            checkpoints.append(episode)
            if getattr(agent, 'has_converged', False):
                break

    return checkpoints, agreements

def run_grid_search(agent_class, env, optimal_policy, param_grid, name):
    results = []

    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        print(f"\n{name}: Testing {params}")

        # Add fixed args
        params.update(dict(
            convergence_tol=1e-6,
            patience=50,
        ))

        try:
            checkpoints, agreements = train_and_evaluate(
                agent_class, params, env, optimal_policy,
                episodes=5000, max_steps=500, eval_interval=100
            )
            final_agreement = agreements[-1] if agreements else 0
            results.append((params.copy(), final_agreement, checkpoints, agreements))
        except Exception as e:
            print(f"Failed with {params}: {e}")

    results.sort(key=lambda x: -x[1])
    print(f"\nTop results for {name}:")
    for params, score, _, _ in results[:5]:
        print(f"{score:.4f} – {params}")
    
    # Visualization of top 5 runs
    plt.figure(figsize=(10, 6))
    for params, _, checkpoints, agreements in results[:5]:
        label = f"{name}: eps={params['epsilon']}, alpha={params['alpha']}, gamma={params['gamma']}"
        plt.plot(checkpoints, agreements, label=label)

    plt.axhline(1.0, linestyle='--', color='gray', label='Optimal Policy')
    plt.xlabel("Episodes")
    plt.ylabel("Policy Agreement")
    plt.title(f"Top 5 {name} Configurations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()