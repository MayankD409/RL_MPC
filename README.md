# Reinforcement Learning with Model Predictive Control

This project implements a reinforcement learning (RL) agent integrated with a Model Predictive Controller (MPC) for autonomous driving simulations using SUMO (Simulation of Urban MObility). The RL agent learns to adjust the weights of the MPC cost function to optimize vehicle control in various traffic scenarios.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
    - [Training the Agent](#training-the-agent)
    - [Simulation Parameters](#simulation-parameters)
- [Files Description](#files-description)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Project Structure

- `train_rl_agent.py`: Main script to train the RL agent using PPO, TD3, or SAC algorithms.
- `mpc_controller.py`: Defines the MPC controller and computes control actions.
- `run_simulation.py`: Manages the simulation environment and runs episodes.
- `models/`:
    - `rl_agent_ppo.py`: PPO agent implementation.
    - `rl_agent_td3.py`: TD3 agent implementation.
    - `rl_agent_sac.py`: SAC agent implementation.
- `requirements.txt`: Python package dependencies.

## Installation

1. **Clone the repository**:
     ```bash
     git clone https://github.com/MayankD409/RL_MPC.git
     cd RL_MPC
     ```

2. **Set up a virtual environment (optional)**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Python dependencies**:
     ```bash
     pip install -r requirements.txt
     ```

4. **Install SUMO**:
     - Download and install SUMO from the [official website](https://www.eclipse.org/sumo/).
     - Set the `SUMO_HOME` environment variable:
         ```bash
         export SUMO_HOME="/path/to/sumo"
         ```

## Usage

### Training the Agent

1. **Select the RL agent** in `train_rl_agent.py`:
     ```python
     agent_type = "TD3"  # Options: "PPO", "SAC", "TD3"
     ```

2. **Run the training script**:
     ```bash
     python train_rl_agent.py
     ```

3. **Monitor training progress**:
     - Training results are saved in the `results/` directory.
     - Plots for rewards, success rates, collision rates, and average jerk are generated.

### Simulation Parameters

- **Simulation scenarios** are defined in `run_simulation.py` using SUMO configuration files.
- **MPC controller settings** can be adjusted in `mpc_controller.py`.
- **RL agent hyperparameters** are set within each agent's script in the `models/` directory.

## Files Description

- **`train_rl_agent.py`**: Trains the RL agent, handles logging and result plotting.
- **`mpc_controller.py`**: Implements the MPC logic and computes control inputs.
- **`run_simulation.py`**: Sets up the SUMO environment, runs simulations, and collects data.
- **`models/`**:
    - **`rl_agent_ppo.py`**: Proximal Policy Optimization agent.
    - **`rl_agent_td3.py`**: Twin Delayed DDPG agent.
    - **`rl_agent_sac.py`**: Soft Actor-Critic agent.
- **`requirements.txt`**: Contains all the Python libraries required to run the project.

## Dependencies

- Python 3.x
- SUMO (Simulation of Urban MObility)
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym
- CVXPY
- Additional packages listed in `requirements.txt`

## Acknowledgments

- SUMO developers for the simulation tools.
- OpenAI for the RL algorithms.
- PyTorch community for the deep learning framework.
