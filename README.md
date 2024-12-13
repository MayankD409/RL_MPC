# RL-MPC Project

## Overview
This project integrates Reinforcement Learning (RL) with Model Predictive Control (MPC) to create an advanced control system. The goal is to leverage the strengths of both RL and MPC to achieve optimal control performance in dynamic environments.

## Project Structure
- `src/`: Contains the source code for the project.
- `data/`: Directory for storing datasets and results.
- `models/`: Pre-trained models and saved checkpoints.
- `scripts/`: Utility scripts for running experiments and evaluations.
- `docs/`: Documentation and additional resources.

## Requirements
- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- Gym

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/rl_mpc.git
    cd rl_mpc
    ```
2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
1. Train the RL agent:
    ```bash
    python src/train_rl_agent.py
    ```
2. Train the MPC controller:
    ```bash
    python src/train_mpc_controller.py
    ```
3. Run the integrated RL-MPC system:
    ```bash
    python src/run_rl_mpc.py
    ```

## Troubleshooting
- **Issue:** ModuleNotFoundError
  - **Solution:** Ensure all dependencies are installed. Run `pip install -r requirements.txt` again.
- **Issue:** CUDA errors
  - **Solution:** Verify that your CUDA and cuDNN versions are compatible with your TensorFlow version.
- **Issue:** Slow training
  - **Solution:** Check if GPU is being utilized. If not, ensure TensorFlow is configured to use the GPU.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any questions or issues, please open an issue on GitHub or contact the maintainer at your.email@example.com.
