# RL4PG - Reinforcement Learning for Power Grids

This repository contains the code for my Master's thesis project **"Graph-Based Multi-Agent Reinforcement Learning for Power Grid Control"** as part of the Master degree in Computer Science and Engineering at Politecnico di Milano.

The goal of this project is to develop a **scalable and adaptive control framework** for electrical power grids using **Graph Neural Networks (GNNs)** and **Multi-Agent Reinforcement Learning (MARL)**, tested on the realistic simulation environment provided by **Grid2Op**.

## 📚 Thesis Summary

Modern power grids are becoming increasingly complex due to the integration of renewable energy sources and the electrification of energy-intensive sectors. Traditional control strategies are no longer sufficient to handle the dynamic and high-dimensional nature of power grid operations.

This project proposes a novel **graph-based multi-agent RL approach** that:

- **Decomposes both the action and observation space** to enhance scalability.
- Utilizes a **shared GNN** to generate informative, localized observations.
- Employs **Deep Q-Learning from Demonstrations (DQfD)** to improve training stability.
- Integrates **Bootstrapped Reward Shaping (BSRS)** to address diluted rewards.

The framework has been tested on the `l2rpn_case14_sandbox` environment within Grid2Op and demonstrated **superior performance** compared to baseline and much **lower inference time** compare to the simulation-based Expert controller.


---

## 🗂️ Repository Structure

```
RL4PG/
│
├── RL4pg/                        # Main library with all modules
│   ├── Initialize_Env.py        # Set Grid2Op paths here
│   ├── utils.py
│   ├── __init__.py
│   │
│   ├── DeepL/                   # Deep learning modules (GNN, models)
│   │   ├── GraphNN.py
│   │   ├── Models.py
│   │   └── __init__.py
│   │
│   ├── Graph_Processing/        # Graph structure processing and utilities
│   │   ├── GP_Manager.py
│   │   └── __init__.py
│   │
│   └── RL/                      # Reinforcement learning modules
│       ├── Converters.py
│       ├── Environments.py
│       ├── Managers.py
│       ├── PG_Agent.py
│       ├── ReplyBuffers.py
│       ├── Trainers.py
│       ├── __init__.py
│       │
│       └── DeepQL/              # Deep Q-Learning components
│           ├── Agents.py
│           ├── Estimator_Manager.py
│           ├── Policy.py
│           └── Q_estimators.py
│
├── Config files/                # JSON configuration files for experiments
│   └── *.json
│
├── main_DQFD.py                 # Main script to launch training
│
├── README.md                    # This file
│
└── requirements.txt

```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Carlo000ml/RL4PG.git
cd RL4PG
```

### 2. Install dependencies

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

> 📝 Note: You must install [`Grid2Op`](https://grid2op.readthedocs.io/en/latest/install.html) and download the necessary **chronics** for the simulation.

---

### 3. Set up Grid2Op Chronics

After installing Grid2Op and downloading the environment data (e.g., `l2rpn_case14_sandbox`), a folder named `data_grid2op` will be created.

Update the path in `RL4pg/Initialize_Env.py`:

```python
base_path = "/path/to/data_grid2op"
```

---

## 🧪 Running Experiments

### Run an experiment

```bash
python -m main_DQFD --config "Config files/your_config.json"
```

Make sure the specified config file exists and is properly configured.

### Monitor training with TensorBoard

```bash
tensorboard --logdir=path_to_log_file
```

Then open your browser at `http://localhost:6006/` or whatever localhost number tensorboard is using.

---
