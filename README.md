# Data Intelligence Challenge – 2AMC15

This repository contains three reinforcement-learning agents applied to a grid-world delivery task, along with all scripts and instructions to **reproduce the results** presented in the report:

- **Value Iteration** (`agents/value_iteration.py`)  
- **On-Policy Monte Carlo** (`agents/mc_onpolicy.py`)  
- **Q-Learning** (`agents/q_learning.py`)  


## 🔧 Setup

1. **Clone the repo**  
```bash
git clone https://github.com/lorypota/DIC-ReinforcementLearning.git
cd DIC-ReinforcementLearning
```
2. Create & activate your environment with Python >= 3.10
```bash
# with conda
conda create -n dic2025 python=3.11
conda activate dic2025

# or with venv
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt 
```

## 🚀 Usage

### 1. Value Iteration
    ```bash
    python train_value_iteration.py GRID [GRID ...] \
      [--no_gui] \
      [--sigma SIGMA] \
      [--fps FPS] \
      [--iter N_SWEEPS] \
      [--random_seed SEED] \
      [--gamma GAMMA] \
      [--theta TOL] \
      [--patience P]
    ```
- `GRID` : one or more .npy grid configs (e.g. grid_configs/A1_grid.npy)
- `--sigma` : env stochasticity (default 0.1)
- `--iter` : max number of iterations
- `--gamma` : discount factor (default 0.9)
- `--theta` : convergence threshold (default 1e-6)
- `--patience` : sweeps under tol for convergence (default 1)

### 2. On-Policy Monte Carlo

```bash
python train_mc_onpolicy.py GRID [GRID ...] \
  [--no_gui] \
  [--sigma SIGMA] \
  [--fps FPS] \
  [--episodes N_EPISODES] \
  [--iter MAX_STEPS] \
  [--random_seed SEED]
```

- `--episodes` : total episodes to train (default 10000)
- `--iter` : max steps per episode (default 500)

### 3. Q-Learning

```bash
python train_q_learning.py GRID [GRID ...] \
  [--no_gui] \
  [--sigma SIGMA] \
  [--fps FPS] \
  [--episodes N_EPISODES] \
  [--iter MAX_STEPS] \
  [--random_seed SEED]
```

## 📊 Reproducing Experiments

1. **Grid Environment Comparison**  
```bash
# simple grid
python train_value_iteration.py grid_configs/A1_grid.npy --iter 1000 --no_gui
python train_q_learning.py      grid_configs/A1_grid.npy --episodes 5000 --no_gui
python train_mc_onpolicy.py     grid_configs/A1_grid.npy --episodes 5000 --no_gui

# large grid
python train_value_iteration.py grid_configs/large_grid.npy --iter 1000 --no_gui
python train_q_learning.py      grid_configs/large_grid.npy --episodes 5000 --no_gui
python train_mc_onpolicy.py     grid_configs/large_grid.npy --episodes 5000 --no_gui
```

2. Epsilon Decay vs. Fixed
    
   Toggle or comment out the decay lines in each agent’s `__init__()` to compare.

3. Hyperparameter Grid Search
    
    For example you can use the following bash script:
```bash
for eps in 0.01 0.05 0.1; do
  for alpha in 0.01 0.05 0.1; do
    for gamma in 0.85 0.90 0.95; do
      # adjust in-code or via env vars
    done
  done
done
```

4. Stochasticity Sweep
```bash
for sigma in 0.0 0.05 0.1 0.2 0.3 0.4; do
  python train_q_learning.py  grid_configs/large_grid.npy --episodes 5000 --sigma $sigma --no_gui
  python train_mc_onpolicy.py grid_configs/large_grid.npy --episodes 5000 --sigma $sigma --no_gui
done
```

5. First-visit vs Every-visit MC 
In agents/mc_onpolicy.py, set first_visit=True or False, then re-run:
```bash
python train_mc_onpolicy.py grid_configs/large_grid.npy --episodes 5000 --no_gui
```

6. Best Agent Showcase

    Q-Learning with epsilon=0.2, alpha=0.2, gamma=0.95, sigma=0.2
```bash
python train_q_learning.py grid_configs/large_grid.npy --episodes 5000 --sigma 0.2 --random_seed 42 --no_gui
```

## 📝 Notes

- Agents print a convergence message when criteria are met.  
- Final evaluation via `evaluate_agent()` reports reward, steps, and failed moves.  
- To save or visualize plots, enable `show_images=True` or modify `evaluate_agent()`.  
- Feel free to tweak hyperparameters or add new experiments.
