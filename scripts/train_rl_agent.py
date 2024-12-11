import numpy as np
import random
import matplotlib.pyplot as plt
from rl_agent_sac import SACAgent
from run_simulation import run_episode, SCENARIO_CONFIG_FILES
from mpc_controller import MPCController
from tqdm import tqdm

state_dim = 25
action_dim = 3
agent = SACAgent(state_dim, action_dim)
mpc = MPCController()

num_episodes = 200

episode_rewards = []
success_rates = []
collisions_count = []
average_jerks = []
times_to_lane_change = []

print("Note: Negative reward values are not 'bad' per se. They discourage undesirable actions/events. The agent learns to maximize cumulative reward, avoiding negative outcomes.")

for episode in tqdm(range(num_episodes), desc="Training"):
    scenario_config = random.choice(SCENARIO_CONFIG_FILES)
    episode_reward, done, reason = run_episode(mpc, agent, scenario_config)
    episode_rewards.append(episode_reward)

    # Track metrics
    # If reason contains "Collision" -> increment collision count
    collisions_count.append(1 if "Collision" in reason else 0)
    # Success if "StableLaneChange" in reason
    success = 1 if "StableLaneChange" in reason else 0
    success_rates.append(success)
    # Approximate jerk: we have jerk logged each step in run_simulation, but we only have final?
    # For simplicity, let's say we computed avg jerk from logs or store them from run_episode (not implemented fully)
    # Here just append a placeholder 0.5 jerk
    average_jerks.append(0.5)
    # Time to lane change = steps until stable if stable else MAX_STEPS
    # Not exact code, but can track steps. For now, assume step is returned?
    # Let's say we ended at certain step means time is step*0.1 sec
    # For demonstration, just assume max_steps/2
    times_to_lane_change.append(150)  # placeholder

    print(f"Episode {episode}: Reward={episode_reward:.2f}, Reason={reason}")

    if (episode+1) % 5 == 0:
        agent.save_model("sac_model.pth")

# Plot episode rewards
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Training Rewards Over Episodes")
plt.savefig("training_rewards.png")
plt.close()

# Plot success rate (rolling average)
plt.figure()
plt.plot(np.cumsum(success_rates)/np.arange(1,len(success_rates)+1))
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Success Rate Over Episodes")
plt.savefig("success_rate.png")
plt.close()

# Similarly plot collisions_count over episodes (cumulative or rolling avg)
plt.figure()
plt.plot(np.cumsum(collisions_count)/np.arange(1,len(collisions_count)+1))
plt.xlabel("Episode")
plt.ylabel("Collision Rate")
plt.title("Collision Rate Over Episodes")
plt.savefig("collision_rate.png")
plt.close()

# Plot average_jerks (placeholder)
plt.figure()
plt.plot(average_jerks)
plt.xlabel("Episode")
plt.ylabel("Average Jerk (Placeholder)")
plt.title("Comfort Metric (Jerk) Over Episodes")
plt.savefig("jerk_over_episodes.png")
plt.close()

# Plot times_to_lane_change (placeholder)
plt.figure()
plt.plot(times_to_lane_change)
plt.xlabel("Episode")
plt.ylabel("Time to Lane Change (Placeholder)")
plt.title("Efficiency Metric Over Episodes")
plt.savefig("time_to_lane_change.png")
plt.close()
