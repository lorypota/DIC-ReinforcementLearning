# world/environment.py
import random
import datetime
import numpy as np
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from world.helpers import save_results, action_to_direction

try:
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    from os import path, pardir
    import sys
    root_path = path.abspath(path.join(path.join(path.abspath(__file__), pardir), pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path

class Environment:
    def __init__(self,
                 grid_fp: Path,
                 no_gui: bool = False,
                 sigma: float = 0.,
                 agent_start_pos: tuple[int, int] = None,
                 reward_fn: callable = None,
                 target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = 0):
        """Creates the Grid Environment for the Reinforcement Learning robot."""
        random.seed(random_seed)
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        self.grid_fp = grid_fp
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None
        # Initialize the grid by calling reset
        self.reset()

    def _reset_info(self) -> dict:
        return {"target_reached": False, "agent_moved": False, "actual_action": None}

    @staticmethod
    def _reset_world_stats() -> dict:
        return {"cumulative_reward": 0, "total_steps": 0, "total_agent_moves": 0,
                "total_failed_moves": 0, "total_targets_reached": 0}

    def _initialize_agent_pos(self):
        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1])
            if self.grid[pos] == 0:
                self.agent_pos = pos
            else:
                raise ValueError("Attempted to place agent on top of obstacle, delivery location or charger")
        else:
            warn("No initial agent positions given. Randomly placing agents on the grid.")
            zeros = np.where(self.grid == 0)
            idx = random.randint(0, len(zeros[0]) - 1)
            self.agent_pos = (zeros[0][idx], zeros[1][idx])

    def reset(self, **kwargs) -> tuple[int, int]:
        for k, v in kwargs.items():
            match k:
                case "grid_fp": self.grid_fp = v
                case "agent_start_pos": self.agent_start_pos = v
                case "no_gui": self.no_gui = v
                case "target_fps": self.target_spf = 1. / v
                case _: raise ValueError(f"{k} is not one of the possible keyword arguments.")
        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()
        if not self.no_gui:
            self.gui = GUI(self.grid.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()
        return self.agent_pos

    def _move_agent(self, new_pos: tuple[int, int]):
        match self.grid[new_pos]:
            case 0:
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case 1 | 2:
                self.world_stats["total_failed_moves"] += 1
                self.info["agent_moved"] = False
            case 3:
                self.agent_pos = new_pos
                self.grid[new_pos] = 0
                if np.sum(self.grid == 3) == 0:
                    self.terminal_state = True
                self.info["target_reached"] = True
                self.world_stats["total_targets_reached"] += 1
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
            case _:
                raise ValueError(f"Grid is badly formed. It has a value of {self.grid[new_pos]} at position {new_pos}.")

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.world_stats["total_steps"] += 1
        is_single_step = False
        if not self.no_gui:
            start_time = time()
            while self.gui.paused:
                if self.gui.step:
                    is_single_step = True
                    self.gui.step = False
                    break
                paused_info = self._reset_info()
                paused_info["agent_moved"] = True
                self.gui.render(self.grid, self.agent_pos, paused_info, 0, is_single_step)
        val = random.random()
        if val > self.sigma:
            actual_action = action
        else:
            actual_action = random.randint(0, 3)
        self.info["actual_action"] = actual_action
        direction = action_to_direction(actual_action)
        new_pos = (self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1])
        reward = self.reward_fn(self.grid, new_pos)
        print(f"Step {self.world_stats['total_steps']}: Action {action}, New Pos {new_pos}, Reward {reward}")  # Debug
        self._move_agent(new_pos)
        self.world_stats["cumulative_reward"] += reward
        if not self.no_gui:
            time_to_wait = self.target_spf - (time() - start_time)
            if time_to_wait > 0:
                sleep(time_to_wait)
            self.gui.render(self.grid, self.agent_pos, self.info, reward, is_single_step)
        return self.agent_pos, reward, self.terminal_state, self.info

    @staticmethod
    def _default_reward_function(grid, agent_pos) -> float:
        match grid[agent_pos]:
            case 0: reward = -1
            case 1 | 2: reward = -5
            case 3: reward = 10
            case _: raise ValueError(f"Grid cell should not have value: {grid[agent_pos]} at position {agent_pos}")
        return reward

    @staticmethod
    def evaluate_agent(grid_fp: Path, agent: BaseAgent, max_steps: int, sigma: float = 0.,
                       agent_start_pos: tuple[int, int] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       show_images: bool = False):
        env = Environment(grid_fp=grid_fp, no_gui=True, sigma=sigma,
                          agent_start_pos=agent_start_pos, target_fps=-1, random_seed=random_seed)
        state = env.reset()
        initial_grid = np.copy(env.grid)
        agent_path = [env.agent_pos]
        for _ in trange(max_steps, desc="Evaluating agent"):
            action = agent.take_action(state)
            state, _, terminated, _ = env.step(action)
            agent_path.append(state)
            if terminated:
                break
        env.world_stats["targets_remaining"] = np.sum(env.grid == 3)
        path_image = visualize_path(initial_grid, agent_path)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        save_results(file_name, env.world_stats, path_image, show_images)

    def get_grid(self) -> np.ndarray:
        """Returns the current grid state."""
        return self.grid