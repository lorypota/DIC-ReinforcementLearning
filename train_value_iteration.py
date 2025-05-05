from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange

try:
    from world import Environment
    from agents.value_iteration import ValueIterationAgent
except ModuleNotFoundError:
    from os import path, pardir, sys
    root = path.abspath(path.join(path.abspath(__file__), pardir, pardir))
    if root not in sys.path:
        sys.path.append(root)
    from world import Environment
    from agents.value_iteration import ValueIterationAgent


def parse_args():
    p = ArgumentParser(description="DIC RL Trainer")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Grid file(s) to use.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disable GUI.")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Environment stochasticity.")
    p.add_argument("--fps", type=int, default=30,
                   help="FPS if GUI on.")
    p.add_argument("--iter", type=int, default=500,
                   help="Number of interaction steps.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed.")
    p.add_argument("--gamma", type=float, default=0.9,
                   help="Discount for VI agent.")
    p.add_argument("--theta", type=float, default=1e-6,
                   help="Convergence tol for VI.")
    p.add_argument("--patience", type=int, default=1,
                   help="Number of consecutive sweeps under tol to declare convergence.")
    return p.parse_args()


def main(grid_paths, no_gui, iters, fps, sigma, random_seed, gamma, theta, patience):
    for grid_fp in grid_paths:
        env = Environment(grid_fp, no_gui, sigma=sigma,
                          target_fps=fps, random_seed=random_seed)
        state = env.reset(agent_start_pos=env.agent_start_pos)

        agent = ValueIterationAgent(str(grid_fp), sigma=sigma,
                                    gamma=gamma, theta=theta,
                                    patience=patience)

        # Interaction loop with convergence check
        for _ in trange(iters, desc="Running agent"):
            action = agent.take_action(state)
            state, reward, terminated, info = env.step(action)
            agent.update(state, reward, info.get("actual_action", action))

            if agent.has_converged:
                break

            if terminated:
                state = env.reset(agent_start_pos=env.agent_start_pos)

        # Evaluate final policy
        Environment.evaluate_agent(
            grid_fp,
            agent,
            max_steps=iters,
            sigma=sigma,
            agent_start_pos=env.agent_start_pos,
            random_seed=random_seed,
            show_images=False
        )

if __name__ == '__main__':
    args = parse_args()
    main(
        args.GRID,
        args.no_gui,
        args.iter,
        args.fps,
        args.sigma,
        args.random_seed,
        args.gamma,
        args.theta,
        args.patience
    )
