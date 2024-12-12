import numpy as np
import random
import matplotlib.pyplot as plt
from run_simulation import run_episode
from mpc_controller import MPCController

# Choose which agent to evaluate
agent_type = "PPO"
state_dim = 25 
action_dim = 3

# Load the appropriate agent
if agent_type == "PPO":
    from rl_agent_ppo import PPOAgent
    agent = PPOAgent(state_dim, action_dim)
    agent.load_model("model.pth")  # Ensure you implemented load_model in PPOAgent
elif agent_type == "TD3":
    from rl_agent_td3 import TD3Agent
    agent = TD3Agent(state_dim, action_dim)
    agent.load_model("model.pth")
else:
    from rl_agent_sac import SACAgent
    agent = SACAgent(state_dim, action_dim)
    agent.load_model("model.pth")

mpc = MPCController()

# Evaluation scenarios
SCENARIO_CONFIG_FILES = [
    "config/sumocfg/seven_lane_light.sumocfg",
    "config/sumocfg/seven_lane_medium.sumocfg",
    "config/sumocfg/seven_lane_heavy.sumocfg"
]

num_eval_episodes = 10
rewards = []
successes = 0
collisions = 0

time_to_stable_values = []
jerks = []

for i in range(num_eval_episodes):
    scenario = random.choice(SCENARIO_CONFIG_FILES)
    ep_reward, done, reason, avg_jerk, stable_lane_step = run_episode(mpc, agent, scenario)
    rewards.append(ep_reward)
    if "StableLaneChange" in reason:
        successes += 1
    if "Collision" in reason:
        collisions += 1
    jerks.append(avg_jerk)
    time_to_stable_values.append(stable_lane_step*0.1)  # Convert steps to seconds if step length=0.1s

    print(f"Eval Episode {i}: Reward={ep_reward:.2f}, Reason={reason}, AvgJerk={avg_jerk:.4f}, TimeToStable={stable_lane_step*0.1:.2f}s")

success_rate = successes / num_eval_episodes
collision_rate = collisions / num_eval_episodes
avg_reward = np.mean(rewards)
avg_jerk_val = np.mean(jerks)
avg_time_to_stable = np.mean(time_to_stable_values)

print("\nEvaluation Results:")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Success Rate: {success_rate:.2f}")
print(f"Collision Rate: {collision_rate:.2f}")
print(f"Average Jerk: {avg_jerk_val:.4f}")
print(f"Average Time to Stable: {avg_time_to_stable:.2f}s")

# Optional: plot evaluation metrics
plt.figure()
plt.boxplot([rewards], labels=[agent_type])
plt.title("Evaluation Episode Rewards")
plt.savefig("eval_rewards.png")
plt.close()

# You can add more plots if desired
plt.figure()
plt.hist(time_to_stable_values, bins=5)
plt.title("Distribution of Time to Stable Lane")
plt.xlabel("Time (s)")
plt.ylabel("Count")
plt.savefig("eval_time_to_stable.png")
plt.close()
