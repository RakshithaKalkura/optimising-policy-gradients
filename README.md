# Deep Reinforcement Learning: Optimising Policy Gradients.

This repository contains implementations of various policy gradient algorithms for Deep Reinforcement Learning (DRL), including REINFORCE, PPO, TRPO, and A2C. The experiments are conducted on the LunarLander-v2 environment from OpenAI Gym.

## Repository Structure

- **README.md**: Project overview and usage instructions.
- **requirements.txt**: Python dependencies.
- **main.py**: Entry point for running experiments.
- **configs/**: Contains configuration files (if applicable).
- **src/**: Source code for the algorithms and utility functions:
  - `environment.py`: Environment setup.
  - `policy_network.py`: Policy network definition.
  - `train_reinforce.py`: Implementation of REINFORCE.
  - `train_ppo.py`: Implementation of PPO.
  - `train_trpo.py`: (Stub) Implementation of TRPO.
  - `train_a2c.py`: Implementation of A2C.
  - `utils.py`: Utility functions (e.g., logging, plotting).
- **logs/** and **results/**: Directories for storing logs and result plots.

## Usage

Run the main script with the desired algorithm and optimizer. For example, to run REINFORCE with Adam:

```bash
python main.py --algorithm reinforce --optimizer adam


