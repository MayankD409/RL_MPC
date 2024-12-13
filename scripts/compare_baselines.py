import numpy as np
import random
import matplotlib.pyplot as plt
from run_simulation import run_episode
from mpc_controller import MPCController
from models.rl_agent_ppo import PPOAgent
from models.rl_agent_td3 import TD3Agent
from models.rl_agent_sac import SACAgent

# Dummy baseline agent with fixed weights
class FixedWeightAgent:
    def get_action(self, state):
        return 1.0, 1.0, 1.0
    def store_transition(self,s,a,r,ns,d): pass
    def update_networks(self): pass
    def update_networks_end_episode(self): pass

def run_multiple_episodes(agent, mpc, scenario_list, num_episodes=10):
    rewards = []
    success_counts = 0
    collisions = 0
    for _ in range(num_episodes):
        scenario = random.choice(scenario_list)
        ep_reward, done, reason, avg_jerk, stable_lane_step = run_episode(mpc, agent, scenario)
        rewards.append(ep_reward)
        if "StableLaneChange" in reason:
            success_counts += 1
        if "Collision" in reason:
            collisions += 1
    success_rate = success_counts/num_episodes
    collision_rate = collisions/num_episodes
    return rewards, success_rate, collision_rate

if __name__ == "__main__":
    # Assume we have a trained RL model for comparison
    # Set agent_type to RL chosen algorithm (PPO, SAC, TD3)
    agent_type = "TD3"
    state_dim = 25
    action_dim = 3

    if agent_type == "PPO":
        agent = PPOAgent(state_dim, action_dim)
        agent.load_model("model.pth")  # if implemented
    elif agent_type == "TD3":
        agent = TD3Agent(state_dim, action_dim)
        agent.load_model("model.pth")
    else:
        agent = SACAgent(state_dim, action_dim)
        agent.load_model("sac_model.pth")

    mpc = MPCController()
    baseline_agent = FixedWeightAgent()

    num_test_episodes = 10
    SCENARIO_CONFIG_FILES = [
        "config/sumocfg/seven_lane_light.sumocfg",
        "config/sumocfg/seven_lane_medium.sumocfg",
        "config/sumocfg/seven_lane_heavy.sumocfg"
    ]

    # Baseline
    baseline_rewards, baseline_success, baseline_collision = run_multiple_episodes(baseline_agent, mpc, SCENARIO_CONFIG_FILES, num_test_episodes)

    # RL
    rl_rewards, rl_success, rl_collision = run_multiple_episodes(agent, mpc, SCENARIO_CONFIG_FILES, num_test_episodes)

    # Compare rewards
    plt.figure()
    plt.boxplot([baseline_rewards, rl_rewards], labels=["Baseline", "RL+MPC"])
    plt.title("Reward Comparison")
    plt.ylabel("Episode Reward")
    plt.savefig("compare_rewards.png")
    plt.close()

    print("Baseline:", "Success:", baseline_success, "Collision:", baseline_collision)
    print("RL+MPC:", "Success:", rl_success, "Collision:", rl_collision)

    # Bar chart for success and collision
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].bar(["RL+MPC","Baseline"], [baseline_success, rl_success])
    ax[0].set_title("Success Rate")
    ax[0].set_ylim([0,1])
    ax[1].bar(["Baseline","RL+MPC"], [baseline_collision, rl_collision])
    ax[1].set_title("Collision Rate")
    ax[1].set_ylim([0,1])
    plt.savefig("compare_rates.png")
    plt.close()
