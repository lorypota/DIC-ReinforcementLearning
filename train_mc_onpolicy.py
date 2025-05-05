"""
Train your RL Agent in this file.
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.mc_onpolicy import MonteCarloOnPolicyAgent
except ModuleNotFoundError:
    from os import path, pardir
    import sys

    root_path = path.abspath(path.join(
        path.abspath(__file__), pardir, pardir
    ))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from world import Environment
    from agents.mc_onpolicy import MonteCarloOnPolicyAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at.")
    p.add_argument("--episodes", type=int, default=10_000,
                   help="Number of episodes to train the agent.")
    p.add_argument("--iter", type=int, default=500,
                   help="Maximum number of steps per episode.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, episodes: int, max_steps: int,
         fps: int, sigma: float, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed)

        # Initialize agent
        agent = MonteCarloOnPolicyAgent(
            grid_fp=grid,
            gamma=0.9,
            epsilon=0.1,
            alpha=0.01,
            max_episode_len=max_steps,
            convergence_tol=1e-6,
            patience=10
        )

        # Run training over episodes
        for _ in trange(episodes, desc="Training episodes"):
            if agent.has_converged:
                break

            state = env.reset()
            for _ in range(max_steps):
                chosen_action = agent.take_action(state)

                next_state, reward, terminated, info = env.step(chosen_action)

                actual_action = info.get("actual_action", chosen_action)
                agent.update(
                    prev_state=state,
                    action=actual_action,
                    reward=reward,
                    done=terminated
                )

                state = next_state

                if terminated or agent.has_converged:
                    break

        # Evaluate the agent
        Environment.evaluate_agent(
            grid, agent, max_steps, sigma,
            random_seed=random_seed
        )


if __name__ == '__main__':
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.episodes,
        args.iter,
        args.fps,
        args.sigma,
        args.random_seed
    )
