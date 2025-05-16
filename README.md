# Data Intelligence Challenge – 2AMC15

The repository provides two sets of entry points:

1. **Training & evaluation scripts** at the root:
   - `train_value_iteration.py`
   - `train_mc_onpolicy.py`
   - `train_q_learning.py`

2. **Automated experiment scripts** under `experiments/`:
   - `experiments/grid_search.py`
   - `experiments/sweep_stochasticity.py`
   - `experiments/every_vs_first_visit_MC.py`
   - `experiments/visualize.py`

Make sure you run all commands from the project root (where `requirements.txt` lives) and that your `dic2025` environment is active.

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

2. **Epsilon Decay vs. Fixed**
    
   Toggle or comment out the decay lines in each agent’s `__init__()` to compare.

3. **Hyperparameter Grid Search**
```bash
cd experiments && python grid_search.py
```
- Searches over ε, α, γ for MC and Q-Learning on A1_grid.npy.
- Prints top-5 parameter sets and plots policy agreement curves.

4. **Stochasticity Sweep**
```bash
cd experiments && python sweep_stochasticity.py
```
- Varies σ ∈ {0.0, 0.05, …, 0.4} on A1_grid.npy.
- Plots final policy agreement vs. σ for both agents.

5. **First-visit vs Every-visit MC**
```bash
cd experiments && python every_vs_first_visit_MC.py
```
- Compares `first_visit=True` vs. `False` on large_grid.npy.
- Shows policy agreement over episodes.

6. **Learning Rate Decay Comparison**
```bash
cd experiments && python compare_learning_rates.py
```
- Compares exponential learning rate decay schedules across different bases (e.g., 0.995, 0.99, 0.97, 0.95, 0.93)
- Evaluates both Q-Learning and On-Policy Monte Carlo on A1_grid.npy
- Plots policy agreement vs. training episodes for each decay schedule



## 📝 Notes

- All experiment scripts assume default hyperparameters as in the report; edit the top of each script to tweak.  
- Scripts use `matplotlib` to display figures; close each plot window to proceed.  
- Convergence messages and final metrics still appear in the console via `evaluate_agent()`.  
- Feel free to integrate new experiments by following the pattern in `experiments/helper.py`.  
