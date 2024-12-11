from rl_agent_sac import SACAgent
from run_simulation import run_episode
from mpc_controller import MPCController

SCENARIO_CONFIG_FILES = [
    "config/sumocfg/seven_lane_heavy.sumocfg",
    "config/sumocfg/seven_lane_heavy.sumocfg",
    "config/sumocfg/seven_lane_heavy.sumocfg"
]

state_dim = 25
action_dim = 3
agent = SACAgent(state_dim, action_dim)
agent.load_model("sac_model.pth")

mpc = MPCController()

scenario_config = "config/sumocfg/seven_lane_heavy.sumocfg"
episode_reward, done = run_episode(mpc, agent, scenario_config)
print("Evaluation Reward:", episode_reward)
