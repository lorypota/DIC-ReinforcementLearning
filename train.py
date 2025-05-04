# train.py
"""
Train your RL Agent in this file.
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import numpy as np

try:
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.value_iteration_agent import ValueIterationAgent
    from agents.monte_carlo_agent import MonteCarloAgent
    from agents.q_learning_agent import QLearningAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys
    root_path = path.abspath(path.join(path.join(path.abspath(__file__), pardir), pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.value_iteration_agent import ValueIterationAgent
    from agents.monte_carlo_agent import MonteCarloAgent
    from agents.q_learning_agent import QLearningAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+", help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true", help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1, help="Stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30, help="Frames per second to render at.")
    p.add_argument("--iter", type=int, default=1000, help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0, help="Random seed value for the environment.")
    p.add_argument("--agent_type", type=str, default="random", choices=["random", "value_iteration", "monte_carlo", "q_learning"],
                   help="Type of agent to train.")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    p.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy policy.")
    p.add_argument("--alpha", type=float, default=0.1, help="Learning rate for Q-Learning.")
    p.add_argument("--max_episode_length", type=int, default=100, help="Max episode length for Monte-Carlo.")
    return p.parse_args()

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int, sigma: float, random_seed: int,
         agent_type: str, gamma: float, epsilon: float, alpha: float, max_episode_length: int):
    for grid in grid_paths:
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps, random_seed=random_seed)
        start_pos = (3, 11) if "A1_grid.npy" in str(grid) else None
        env.reset(agent_start_pos=start_pos)  # Pass start_pos to reset
        
        grid_data = env.get_grid()
        if agent_type == "random":
            agent = RandomAgent()
        elif agent_type == "value_iteration":
            agent = ValueIterationAgent(grid_data, gamma=gamma)
        elif agent_type == "monte_carlo":
            agent = MonteCarloAgent(grid_data, gamma=gamma, epsilon=epsilon, max_episode_length=max_episode_length)
        elif agent_type == "q_learning":
            agent = QLearningAgent(grid_data, gamma=gamma, alpha=alpha, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Training loop with metrics
        returns = []
        episode_rewards = []
        for episode in trange(iters // max_episode_length if agent_type == "monte_carlo" else 1, desc="Episodes"):
            state = env.reset()
            episode_reward = 0
            for t in range(max_episode_length if agent_type == "monte_carlo" else iters):
                action = agent.take_action(state)
                state, reward, terminated, info = env.step(action)
                agent.update(state, reward, info["actual_action"])
                episode_reward += reward
                if terminated or (agent_type != "monte_carlo" and t == iters - 1):
                    break
            if agent_type == "monte_carlo":
                agent.end_episode()
            returns.append(episode_reward)
            episode_rewards.append(episode_reward)

        # Evaluate the agent
        Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)
        
        # Print evaluation metrics
        print(f"\nGrid: {grid}")
        print(f"Agent: {agent_type}")
        print(f"Average Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        if agent_type != "value_iteration":
            q_diff = np.max(np.abs(agent.Q)) - np.min(np.abs(agent.Q))
            print(f"Q-Value Spread (indicating convergence): {q_diff:.2f}")

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed,
         args.agent_type, args.gamma, args.epsilon, args.alpha, args.max_episode_length)