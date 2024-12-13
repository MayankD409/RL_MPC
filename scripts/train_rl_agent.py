import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from run_simulation import run_episode, SCENARIO_CONFIG_FILES
from mpc_controller import MPCController
from models.rl_agent_ppo import PPOAgent
from models.rl_agent_td3 import TD3Agent
from models.rl_agent_sac import SACAgent

# Choose agent type:
agent_type = "TD3"  # or "SAC" or "TD3"

state_dim = 25
action_dim = 3

if agent_type == "PPO":
    agent = PPOAgent(state_dim, action_dim)
elif agent_type == "TD3":
    agent = TD3Agent(state_dim, action_dim)
else:
    agent = SACAgent(state_dim, action_dim)

mpc = MPCController()

num_episodes = 100
results_folder = f"results/{num_episodes}_{agent_type}"

# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

episode_rewards = []
success_rates = []
collisions_count = []
average_jerks = []
time_to_stable_list = []

print("Note: Negative rewards guide policy away from bad states. Adjusted reward function provides incremental rewards.")

# Using tqdm to wrap the range of episodes
for episode in tqdm(range(num_episodes), desc="Training Progress"):
    # Progressive scenario difficulty
    if episode < 100:
        scenario_config = "config/sumocfg/seven_lane_light.sumocfg"
    elif episode < 300:
        scenario_config = "config/sumocfg/seven_lane_medium.sumocfg"
    else:
        scenario_config = "config/sumocfg/seven_lane_heavy.sumocfg"

    ep_reward, done, reason, avg_jerk, stable_lane_step = run_episode(mpc, agent, scenario_config)
    episode_rewards.append(ep_reward)

    # Track success
    success = 1 if "StableLaneChange" in reason else 0
    success_rates.append(success)

    # Track collisions
    collision = 1 if "Collision" in reason else 0
    collisions_count.append(collision)

    # Jerk and stable lane time
    average_jerks.append(avg_jerk)
    time_to_stable_list.append(stable_lane_step * 0.1)  # step * 0.1 (assuming step length=0.1s)

    print(f"Episode {episode}: Reward={ep_reward:.2f}, Reason={reason}, AvgJerk={avg_jerk:.4f}, StableTime={stable_lane_step * 0.1:.2f}s")

    # If SAC or TD3, updates happen inside run_episode or after steps anyway
    # If PPO, we have already updated after run_episode if needed

    if (episode + 1) % 50 == 0:
        # If agent supports model saving
        if hasattr(agent, 'save_model'):
            agent.save_model(os.path.join(results_folder, "model.pth"))

# Plotting results
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Training Rewards Over Episodes")
plt.savefig(os.path.join(results_folder, "training_rewards.png"))
plt.close()

plt.figure()
plt.plot(np.cumsum(success_rates) / np.arange(1, len(success_rates) + 1))
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Success Rate Over Episodes")
plt.savefig(os.path.join(results_folder, "success_rate.png"))
plt.close()

plt.figure()
plt.plot(np.cumsum(collisions_count) / np.arange(1, len(collisions_count) + 1))
plt.xlabel("Episode")
plt.ylabel("Collision Rate")
plt.title("Collision Rate Over Episodes")
plt.savefig(os.path.join(results_folder, "collision_rate.png"))
plt.close()

plt.figure()
plt.plot(average_jerks)
plt.xlabel("Episode")
plt.ylabel("Average Jerk")
plt.title("Average Jerk Over Episodes")
plt.savefig(os.path.join(results_folder, "jerk_over_episodes.png"))
plt.close()

plt.figure()
plt.plot(time_to_stable_list)
plt.xlabel("Episode")
plt.ylabel("Time to Stable Lane (s)")
plt.title("Time to Stable Lane Over Episodes")
plt.savefig(os.path.join(results_folder, "time_to_stable.png"))
plt.close()

print(f"Results saved in folder: {results_folder}")
