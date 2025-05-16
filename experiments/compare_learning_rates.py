import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
import matplotlib.pyplot as plt

from world.environment import Environment
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.mc_onpolicy import MonteCarloOnPolicyAgent

from experiments.helper import (
    extract_policy_from_vi_agent,
    train_and_evaluate
)

def main():
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

    decay_bases = [0.995, 0.990, 0.970, 0.950, 0.930]
    agent_classes = {
        "Q-Learning": QLearningAgent,
        "MC-OnPolicy": MonteCarloOnPolicyAgent
    }

    for name, Agent in agent_classes.items():
        plt.figure(figsize=(10, 6))
        for decay_base in decay_bases:
            base = decay_base
            def lr_schedule(ep, base=base):
                return max(0.0001, 1.0 * (base ** ep))

            class AgentWithDecay(Agent):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.ep_num = 0
                    self.step_counter = 0

                def update(self, *args, **kwargs):
                    if self.step_counter == 0:
                        self.alpha = lr_schedule(self.ep_num)
                    self.step_counter += 1
                    if kwargs.get("done", False):
                        self.ep_num += 1
                        self.step_counter = 0
                    super().update(*args, **kwargs)

            print(f"\\n{name}: decay base={decay_base}")
            checkpoints, agreements = train_and_evaluate(
                agent_class=AgentWithDecay,
                agent_kwargs={
                    "alpha": 1.0,
                    "gamma": 0.9,
                    "epsilon": 0.1,
                    "patience": 30
                },
                env=env,
                optimal_policy=optimal_policy,
                episodes=1000,
                max_steps=300,
                eval_interval=20
            )
            plt.plot(checkpoints, agreements, label=f"decay base={decay_base}")
        
        plt.title(f"{name} – Exponential Decay LR (Small Bases)")
        plt.xlabel("Episodes")
        plt.ylabel("Policy Agreement")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{name.lower().replace('-', '_')}_expdecay_small_extended.png")
        plt.show()

if __name__ == "__main__":
    main()
