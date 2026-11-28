# 🚀 Getting Started

This section explains how to install the project and run the interactive training, gameplay, and evaluation menu.

## 1. Requirements

- Python 3.11 (recommended)
- Windows, macOS, or Linux
- Internet connection to install dependencies
- A graphical desktop environment for gameplay demonstrations

## 2. Extract the Project

Extract the submitted ZIP file and open the `IT493-Space-Invaders-RL` project folder in VS Code or a terminal.

**Important:** Run the commands below from the project root — the folder containing `run_project.py` and `requirements.txt`.

## 3. Create a Virtual Environment

```powershell
python -m venv venv
```

## 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

## 5. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Run the Project

From the project root, run:

```powershell
python run_project.py
```

The interactive menu provides the following options:

| Option | Function | Description |
|---|---|---|
| 1 | Train DQN | Train an extended DQN model for 500,000 timesteps |
| 2 | Train PPO | Train an extended PPO model for 500,000 timesteps |
| 3 | Watch DQN | Watch the trained DQN agent play Space Invaders |
| 4 | Watch PPO | Watch the trained PPO agent play Space Invaders |
| 5 | Evaluate and Compare | Evaluate both agents over 10 episodes and compare their rewards |
| 0 | Exit | Close the program |

---

# 🎮 A Comparative Study of DQN and PPO for Pixel-Based Reinforcement Learning with Saliency Map Analysis

A research project comparing Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) on Space Invaders using raw pixel observations, with saliency map visualizations to analyze learned visual attention.

This project investigates sample efficiency, performance stability, and learned visual attention across three random seeds using Gymnasium, Stable-Baselines3, and PyTorch — all running on CPU with structured experiment tracking.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?logo=openai&logoColor=white&style=for-the-badge)
![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-FF6F00?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge)
![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow&logoColor=white&style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white&style=for-the-badge)

---

## 🧩 System Design

This diagram represents the high-level architecture of the experiment pipeline, including environment preprocessing, training, evaluation, and saliency analysis.

<p align="center">
  <img src="./Readme-Photos/System-Design.png" alt="System Design Diagram" width="800"/>
</p>

### Overview

- ALE/SpaceInvaders-v5 provides raw pixel observations
- Preprocessing pipeline:
  - Grayscale conversion → Resize to 84×84 → Frame stacking (4 frames) → Reward clipping
- Stable-Baselines3 wraps the environment with `VecTransposeImage` for CNN compatibility
- DQN and PPO train independently with identical preprocessing and budgets
- Training callbacks log episode rewards, lengths, timesteps, and elapsed time to CSV
- TensorBoard records algorithm-specific training metrics
- Evaluation callbacks run every 20,000 steps and save the best model
- Checkpoints save every 25,000 steps (DQN also saves replay buffer)
- Saliency maps visualize which pixels influence agent decisions

This design ensures both algorithms are compared under identical conditions for fair research.

---

## 🚀 Features

- Atari Space Invaders environment with full preprocessing pipeline
- Grayscale conversion, 84×84 resizing, 4-frame stacking, reward clipping
- DQN with experience replay buffer and target network
- PPO with clipped surrogate objective and multiple epochs
- CNN-based policies (CnnPolicy) for both algorithms
- 3 random seeds per algorithm (42, 123, 2026)
- 100,000 timestep pilot budget per run
- Evaluation every 20,000 steps with 3 episodes
- Checkpointing every 25,000 steps
- CSV logging for research data
- TensorBoard integration for monitoring
- Saliency map generation for visual attention analysis
- CPU-only execution (GPU optional)
- Structured experiment auditing

---

### 🏗 Folder Structure

```
IT493-Space-Invaders-RL/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dqn/
│   │   ├── config.py
│   │   └── train.py
│   │
│   ├── ppo/
│   │   ├── config.py
│   │   └── train.py
│   │
│   ├── environment/
│   │   ├── env_factory.py
│   │   └── test_preprocessing.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── audit_experiments.py
│   │
│   ├── visualization/
│   │   └── saliency.py
│   │
│   └── callbacks/
│       └── training_logger.py
│
├── models/
│   ├── dqn/
│   │   ├── dqn_seed_42_final.zip
│   │   ├── dqn_seed_123_final.zip
│   │   └── dqn_seed_2026_final.zip
│   │
│   └── ppo/
│       ├── ppo_seed_42_final.zip
│       ├── ppo_seed_123_final.zip
│       └── ppo_seed_2026_final.zip
│
├── logs/
│   ├── dqn/
│   │   ├── seed_42/
│   │   ├── seed_123/
│   │   └── seed_2026/
│   │
│   └── ppo/
│       ├── seed_42/
│       ├── seed_123/
│       └── seed_2026/
│
├── results/
│   ├── training/
│   └── evaluation/
│
├── graphs/
│   ├── learning_curves/
│   └── comparisons/
│
├── saliency_maps/
│   ├── dqn/
│   └── ppo/
│
├── gameplay/
│   ├── dqn/
│   └── ppo/
```

---

## ⚙️ Configuration

### Experiment Settings

| Parameter | Value |
|-----------|-------|
| Random seeds | 42, 123, 2026 |
| Environments | 1 |
| Training timesteps | 100,000 (pilot) |
| Evaluation frequency | 20,000 |
| Evaluation episodes | 3 |
| Checkpoint frequency | 25,000 |
| Device | CPU |

### DQN Hyperparameters

```python
learning_rate = 0.0001
buffer_size = 100000
learning_starts = 10000
batch_size = 32
gamma = 0.99
target_update_interval = 1000
exploration_fraction = 0.1
```

### PPO Hyperparameters

```python
learning_rate = 0.0001
n_steps = 512
batch_size = 128
n_epochs = 4
gamma = 0.99
```

Note: PPO uses `n_steps=512` and `batch_size=128` because only one environment is used. This ensures enough collected experiences to form complete minibatches.

---

## 🐳 Environment Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/IT493-Space-Invaders-RL.git
cd IT493-Space-Invaders-RL
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify environment

```bash
python -m src.environment.test_preprocessing
```

---

## 🧪 Testing

### Test preprocessing pipeline

```bash
python -m src.environment.test_preprocessing
```

Expected output:
- Space Invaders launches correctly
- Observations converted to grayscale
- Frames resized to 84×84
- Four frames stacked: shape (4, 84, 84)
- Random actions advance gameplay
- Processed observations visualized

### Test DQN setup

```bash
python -m src.dqn.train --test
```

### Test PPO setup

```bash
python -m src.ppo.train --test
```

Both should confirm:
- `VecTransposeImage` applied automatically
- Observation shape: (4, 84, 84)
- CNN policy initialized
- CPU execution works

---

## 📊 Training

### Train DQN

```bash
python -m src.dqn.train --seed 42
```

### Train PPO

```bash
python -m src.ppo.train --seed 42
```

Each training run:
- Creates a fresh environment
- Initializes a new CNN-based model
- Trains for 100,000 timesteps
- Logs completed episodes to CSV
- Records metrics in TensorBoard
- Saves final model, checkpoints, and best model
- Repeats for each configured seed

---

## 🔁 Training Pipeline

```
Raw Atari Frame (210×160×3)
  ↓
Grayscale Conversion
  ↓
Resize to 84×84
  ↓
Frame Stacking (4 frames)
  ↓
Reward Clipping (-1, 0, 1)
  ↓
VecTransposeImage (channels-first)
  ↓
CNN Policy (DQN or PPO)
  ↓
Action Selection
  ↓
Environment Step
```

---

## 🧠 Key Concepts Implemented

- Deep Q-Network with experience replay
- Target network with periodic updates
- Epsilon-greedy exploration decay
- Proximal Policy Optimization with clipped objective
- Multiple epochs per rollout
- CNN-based feature extraction from pixels
- Frame stacking for motion perception
- Reward clipping for stable learning
- Training callbacks for automatic logging
- Evaluation callbacks for periodic assessment
- Checkpointing for resumable training
- Saliency map generation for interpretability

---

## 📈 Results

### DQN Evaluation Results (Seed 42)

| Timesteps | Mean Score | Std Dev |
|-----------|------------|---------|
| 20,000 | 285.00 | 0.00 |
| 40,000 | 191.67 | 96.87 |
| 60,000 | 211.67 | 18.86 |
| 80,000 | 285.00 | 0.00 |
| 100,000 | 171.67 | 11.79 |

### PPO Evaluation Results (Seed 42)

| Timesteps | Mean Score | Std Dev |
|-----------|------------|---------|
| 20,000 | 285.00 | 0.00 |
| 40,000 | 285.00 | 0.00 |
| 60,000 | 285.00 | 0.00 |
| 80,000 | 285.00 | 0.00 |
| 100,000 | 180.00 | — |

### DQN Final Evaluation Across Seeds

| Seed | Final Mean |
|------|------------|
| 42 | 171.67 |
| 123 | 181.67 |
| 2026 | 81.67 |
| **Overall** | **145.00 ± 55.08** |

### PPO Final Evaluation Across Seeds

| Seed | Final Mean |
|------|------------|
| 42 | 180.00 |
| 123 | 285.00 |
| 2026 | 285.00 |
| **Overall** | **250.00 ± 60.62** |

### Initial Comparison Summary

| Metric | DQN | PPO |
|--------|-----|-----|
| Training time | 429.67 sec | 220.98 sec |
| Average of evaluation means | 229.00 | 264.00 |
| Final evaluation mean | 171.67 | 180.00 |
| Highest evaluation mean | 285 | 285 |

PPO trained approximately 49% faster and maintained more consistent evaluation scores. Both algorithms experienced a decline in final evaluation for seed 42. DQN showed higher variance across seeds.

---

## ⚡ Performance

- DQN pilot: 1,000 timesteps in 0.55 sec (1,627 FPS)
- PPO pilot: 512 timesteps in 1.03 sec (726 FPS)
- Full DQN run: 100,000 timesteps in ~430 sec
- Full PPO run: 100,000 timesteps in ~221 sec
- CPU-only execution with stable memory usage
- CNN processing verified on actual Space Invaders frames

---

## 🔮 Future Improvements

- Extended training budget (1M+ timesteps)
- Additional algorithms (A2C, SAC)
- Frame stacking ablation studies
- Reward shaping experiments
- GPU-accelerated training
- Hyperparameter sweep automation
- Attention map overlays on gameplay
- Comparative saliency analysis between algorithms
- Statistical significance testing across more seeds
- Video recordings of learned gameplay

---

## 📚 What This Project Demonstrates

- Reinforcement learning from raw pixel inputs
- Comparative algorithm evaluation methodology
- Atari preprocessing pipeline implementation
- Experiment tracking and reproducibility
- CNN-based policy learning
- Interpretability through saliency maps
- Multi-seed experimental design
- Structured data logging and auditing
- Research-grade experiment organization

---

## 🏁 Summary

This project provides a rigorous comparison of DQN and PPO for pixel-based reinforcement learning on Space Invaders, with saliency map analysis to understand learned visual attention. Preliminary results suggest PPO trains faster and maintains more consistent performance, while DQN shows higher variance across seeds. All experiments are reproducible, tracked, and structured for research analysis.
