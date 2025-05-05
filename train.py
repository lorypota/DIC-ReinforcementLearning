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
    from agents.vi_agent import ValueIterationAgent
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent
    from agents.vi_agent import ValueIterationAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--agent", type=str, choices=["random", "value_iter"], default="value_iter",
                   help="Agent to train/evaluate.")
    p.add_argument("--iters", type=int, default=1000,
               help="Number of iterations.")
    return p.parse_args()

def custom_reward_fn(grid: np.ndarray, new_pos: tuple[int,int]) -> float:
        cell = grid[new_pos]
        match cell:
            case 0|4:   # empty or starting
                return -1.0
            case 1|2: # wall or obstacle
                return -1.0
            case 3:   # goal
                return 10.0
            case _:
                raise ValueError(f"Unexpected cell value {cell} at {new_pos}")

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_type: str):
    """Main loop of the program."""
    
    for grid in grid_paths: 
        # Set up the environment
        env = Environment(
            grid_fp=grid,
            no_gui=no_gui,
            sigma=sigma,
            reward_fn=custom_reward_fn,
            target_fps=fps,
            random_seed=random_seed,
        )
        
        # Initialize agent
        if agent_type == "random":
            agent = RandomAgent()
        else:
            agent = ValueIterationAgent(env)
        
        # Always reset the environment to initial state
        state = env.reset()
        for _ in trange(iters):
            
            # Agent takes an action based on the latest observation and info.
            action = agent.take_action(state)

            # The action is performed in the environment
            state, reward, terminated, info = env.step(action)
            
            # If the final state is reached, stop.
            if terminated:
                break

            agent.update(state, reward, info["actual_action"])

        # Evaluate the agent
        Environment.evaluate_agent(
            grid,
            agent,
            iters,
            sigma,
            random_seed=random_seed,
            reward_fn=custom_reward_fn
        )


if __name__ == '__main__':
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.iters,
        args.fps,
        args.sigma,
        args.random_seed,
        args.agent
    )