import os
import sys
import math
import random
import csv
import traci
import time
import logging
import numpy as np
from models.rl_agent_ppo import PPOAgent
# Import the MPC controller
from mpc_controller import MPCController

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mpc_simulation.log"),  # Log to file
        # logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

# Create a logger for this script
logger = logging.getLogger("RunSimulation")

# Path to the SUMO configuration files. These scenario files differ by different route files or traffic flows.
SCENARIO_CONFIG_FILES = [
    "config/sumocfg/seven_lane_light.sumocfg",
    "config/sumocfg/seven_lane_medium.sumocfg",
    "config/sumocfg/seven_lane_heavy.sumocfg"
]

# Set a random scenario for diversity
# SELECTED_CONFIG = random.choice(SCENARIO_CONFIG_FILES)
SELECTED_CONFIG = "config/sumocfg/seven_lane_heavy.sumocfg"

# -------------------------------
# Configuration Parameters
# -------------------------------
MAX_STEPS = 5000          # ~300s if step-length=0.1s
STABLE_STEPS_REQUIRED = 50
PROXIMITY_DISTANCE = 20.0
COLLISION_DISTANCE = 1.0
RISK_OCCUPANCY_HIGH = 0.4
RISK_OCCUPANCY_MED = 0.2
RISK_SPEED_LOW = 10.0
RISK_CLOSE_VEHICLE = 10.0

# Lane speed variations
LANE_SPEEDS = {
    "edge_start_end_0": 33.33,   # slow lane
    "edge_start_end_1": 33.33,   # medium speed lane
    "edge_start_end_2": 33.33,  # fast lane
    "edge_start_end_3": 33.33,   # slow lane
    "edge_start_end_4": 33.33,   # medium speed lane
    "edge_start_end_5": 33.33,  # fast lane
    "edge_start_end_6": 33.33
}

# Logging options
LOG_DATA = True
LOG_FILE = "simulation_data_log.csv"

# Track previous ego states for jerk calculation
previous_ego_speed = None
previous_ego_acc = None
previous_ego_time = None

# Track how long we've been in unsafe proximity
unsafe_proximity_count = 0
UNSAFE_PROXIMITY_THRESHOLD = 5  # Steps of continuous unsafe proximity before terminating

def apply_mpc_weights(W_s, W_e, W_c):
    """
    Placeholder function for applying MPC weights.
    Tanmay would call this after deciding on an action.
    Currently, just prints the weights to demonstrate the interface.
    """
    logger.info(f"Applying MPC weights: Safety={W_s:.2f}, Efficiency={W_e:.2f}, Comfort={W_c:.2f}")

def add_ego_vehicle():
    """Add the ego vehicle with random initial conditions."""
    ego_id = "ego"
    initial_lane = random.randint(0, 6)
    # initial_lane = 0
    initial_speed = random.uniform(0, 10)
    target_lane = random.choice([l for l in range(7) if l != initial_lane])
    # target_lane = 5
    logger.debug("Adding ego vehicle with the following parameters:")
    logger.debug(f"Vehicle ID: {ego_id}")
    logger.debug(f"Initial Lane: {initial_lane}")
    logger.debug(f"Initial Speed: {initial_speed}")
    logger.debug(f"Target Lane: {target_lane}")

    # Ensure route exists for ego
    if "ego_route" not in traci.route.getIDList():
        traci.route.add("ego_route", ["edge_start_end"])
        logger.debug("Added 'ego_route' to SUMO routes.")

    traci.vehicle.add(
        vehID=ego_id,
        routeID="ego_route",
        typeID="car",
        depart="0",
        departLane=str(initial_lane),
        departSpeed=str(initial_speed),
        departPos="0"
    )

    # Set the color of the ego vehicle to white
    traci.vehicle.setColor(ego_id, (255, 255, 255))
    logger.info("Ego vehicle added and colored white.")

    return {
        "id": ego_id,
        "initial_lane": initial_lane,
        "target_lane": target_lane,
        "initial_speed": initial_speed
    }

def get_all_lanes():
    # Since we have one edge "edge_start_end" with 7 lanes:
    edge_id = "edge_start_end"
    lane_count = 7  # known from network file
    lanes = [f"{edge_id}_{i}" for i in range(lane_count)]
    return lanes

def set_lane_speeds():
    """Set distinct lane speeds to differentiate lane attributes."""
    for lane_id, speed in LANE_SPEEDS.items():
        try:
            traci.lane.setMaxSpeed(lane_id, speed)
            logger.debug(f"Set max speed for lane '{lane_id}' to {speed} m/s.")
        except traci.exceptions.TraCIException as e:
            logger.error(f"Error setting speed for lane '{lane_id}': {e}")
            pass

def get_lane_info(ego_id):
    """Retrieve lane occupancy, avg speed, and proximity data relative to ego."""
    try:
        ego_lane_id = traci.vehicle.getLaneID(ego_id)
        ego_x, ego_y = traci.vehicle.getPosition(ego_id)
        logger.debug(f"Ego vehicle '{ego_id}' is in lane '{ego_lane_id}' at position ({ego_x}, {ego_y}).")
    except traci.exceptions.TraCIException:
        print(f"Vehicle '{ego_id}' not found in the simulation.")
        logger.error(f"Error retrieving lane ID or position for ego vehicle '{ego_id}': {e}")
        return None

    lanes = get_all_lanes()
    lane_data = {}

    for lane in lanes:
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        occupancy = traci.lane.getLastStepOccupancy(lane)
        avg_speed = traci.lane.getLastStepMeanSpeed(lane)

        proximity_count = 0
        proximity_dists = []
        for v in vehicles:
            if v != ego_id:
                try:
                    x, y = traci.vehicle.getPosition(v)
                    dist = math.sqrt((x - ego_x)**2 + (y - ego_y)**2)
                    if dist < PROXIMITY_DISTANCE:
                        proximity_count += 1
                        proximity_dists.append(dist)
                except traci.exceptions.TraCIException as e:
                    logger.warning(f"Error retrieving position for vehicle '{v}': {e}")
                    continue

        lane_data[lane] = {
            "occupancy": occupancy,
            "avg_speed": avg_speed,
            "proximity_count": proximity_count,
            "proximity_min_dist": min(proximity_dists) if proximity_dists else None
        }

        logger.debug(f"Lane '{lane}': Occupancy={occupancy}, Avg Speed={avg_speed}, "
                         f"Proximity Count={proximity_count}, Min Distance={lane_data[lane]['proximity_min_dist']}")

    return lane_data

def compute_lane_risk(lane_data, ego_lane_id):
    """Compute a discrete risk score for each lane."""
    lane_risks = {}
    for lane, data in lane_data.items():
        score = 0
        occ = data["occupancy"]
        spd = data["avg_speed"]
        p_min = data["proximity_min_dist"]

        # Weighted scoring could be refined further if needed
        if occ >= RISK_OCCUPANCY_HIGH:
            score += 2
        elif occ >= RISK_OCCUPANCY_MED:
            score += 1

        if spd < RISK_SPEED_LOW:
            score += 1

        if p_min is not None and p_min < RISK_CLOSE_VEHICLE:
            score += 1

        # Map score to discrete level
        if score == 0:
            risk_level = 0
        elif score == 1:
            risk_level = 1
        else:
            risk_level = 2

        lane_risks[lane] = risk_level
        logger.debug(f"Computed risk for lane '{lane}': {risk_level}")

    return lane_risks

def check_collision(ego_id):
    """Check if a collision occurred based on proximity."""
    try:
        ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    except traci.exceptions.TraCIException as e:
        logger.error(f"Error retrieving position for ego vehicle '{ego_id}': {e}")
        return True

    vehicles = traci.vehicle.getIDList()
    for v in vehicles:
        if v != ego_id:
            try:
                x, y = traci.vehicle.getPosition(v)
                dist = math.sqrt((x - ego_x)**2 + (y - ego_y)**2)
                if dist < COLLISION_DISTANCE:
                    logger.error(f"Collision detected between ego vehicle '{ego_id}' and vehicle '{v}'.")
                    return True
            except traci.exceptions.TraCIException as e:
                logger.warning(f"Error retrieving position for vehicle '{v}': {e}")
                continue
    return False

def check_lane_change_completion(ego_id, target_lane, stable_count):
    """Check if ego has stabilized in target lane."""
    try:
        current_lane_id = traci.vehicle.getLaneID(ego_id)
        current_lane_index = int(current_lane_id.split("_")[-1])
        logger.debug(f"Ego vehicle '{ego_id}' is currently in lane '{current_lane_id}' (Index: {current_lane_index}).")
    except traci.exceptions.TraCIException as e:
        logger.error(f"Error retrieving lane ID for ego vehicle '{ego_id}': {e}")
        return 0, False

    if current_lane_index == target_lane:
        stable_count += 1
        logger.debug(f"Ego vehicle '{ego_id}' has been in target lane {target_lane} for {stable_count} steps.")
    else:
        stable_count = 0
        logger.debug(f"Ego vehicle '{ego_id}' is not yet in target lane {target_lane}.")

    stable = stable_count >= STABLE_STEPS_REQUIRED
    if stable:
        logger.info(f"Ego vehicle '{ego_id}' has stabilized in target lane {target_lane}.")
    return stable_count, stable

def check_timeout(step):
    """Check if episode exceeded max steps."""
    if step >= MAX_STEPS:
        logger.warning(f"Simulation step {step} reached the maximum limit of {MAX_STEPS}.")
        return True
    return False

def check_safety_violations(lane_data, ego_lane_id):
    """
    Check for prolonged unsafe proximity or other violations.
    For demonstration, if the ego lane's proximity_count is high consistently,
    consider it a violation.
    """
    global unsafe_proximity_count
    ego_lane_data = lane_data.get(ego_lane_id, {})
    if ego_lane_data.get("proximity_count", 0) > 3:
        unsafe_proximity_count += 1
        logger.debug(f"Unsafe proximity count increased to {unsafe_proximity_count}.")
    else:
        unsafe_proximity_count = 0
        logger.debug("Unsafe proximity count reset to 0.")

    # If we exceed the threshold, end episode
    if unsafe_proximity_count >= UNSAFE_PROXIMITY_THRESHOLD:
        logger.warning(f"Unsafe proximity threshold reached ({UNSAFE_PROXIMITY_THRESHOLD} steps).")
        return True
    return False

def compute_comfort_metrics(ego_id, step_time=0.1):
    """
    Compute approximate jerk as a comfort metric.:
    acc = (v_current - v_previous)/delta_t
    jerk = (acc_current - acc_previous)/delta_t
    """
    global previous_ego_speed, previous_ego_acc, previous_ego_time

    try:
        ego_speed = traci.vehicle.getSpeed(ego_id)
        logger.debug(f"Ego vehicle '{ego_id}' speed: {ego_speed} m/s.")
    except traci.exceptions.TraCIException as e:
        logger.error(f"Error retrieving speed for ego vehicle '{ego_id}': {e}")
        return None, None

    # Compute acceleration if previous speed is available
    if previous_ego_speed is not None:
        acc = (ego_speed - previous_ego_speed) / step_time
        logger.debug(f"Ego vehicle '{ego_id}' acceleration: {acc} m/s².")
    else:
        acc = 0.0
        logger.debug(f"Ego vehicle '{ego_id}' acceleration: {acc} m/s² (initial step).")


    # Compute jerk if previous acceleration is available
    if previous_ego_acc is not None:
        jerk = (acc - previous_ego_acc) / step_time
        logger.debug(f"Ego vehicle '{ego_id}' jerk: {jerk} m/s³.")
    else:
        jerk = 0.0
        logger.debug(f"Ego vehicle '{ego_id}' jerk: {jerk} m/s³ (initial step).")

    # Update previous states
    previous_ego_speed = ego_speed
    previous_ego_acc = acc

    return acc, jerk

def randomize_scenario():
    """
    Randomize scenario by choosing a different config file and possibly different routes.
    Since SUMO configuration is started externally, we just selected SELECTED_CONFIG.
    I have created multiple scenario config files with different route distributions.
    """
    logger.info(f"Selected scenario config: {SELECTED_CONFIG}")

def initialize_logging():
    if LOG_DATA:
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "ego_x", "ego_y", "ego_speed", "ego_lane", 
                                "lane_risks", "W_s", "W_e", "W_c", "acc", "jerk"])
            logger.info(f"Initialized CSV log file: {LOG_FILE}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV log file '{LOG_FILE}': {e}")

def log_data(step, ego_state, lane_risks, W_s, W_e, W_c, acc, jerk):
    if LOG_DATA:
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, ego_state["position"][0], ego_state["position"][1],
                                ego_state["speed"], ego_state["lane_id"], lane_risks,
                                W_s, W_e, W_c, acc, jerk])
            logger.debug(f"Logged data for step {step}.")
        except Exception as e:
            logger.error(f"Failed to log data for step {step}: {e}")

def construct_rl_state(ego_state, lane_data, lane_risks):
    logger.debug("Constructing RL state from ego_state and lane_data.")
    max_speed = 33.33
    ego_x, ego_y = ego_state["position"]
    ego_speed = ego_state["speed"]/max_speed
    current_lane_id = ego_state["lane_id"]
    current_lane_index = int(current_lane_id.split("_")[-1])
    target_lane = ego_state["target_lane"]

    lanes = get_all_lanes()
    lane_features = []
    for lane in lanes:
        ld = lane_data[lane]
        occ = ld["occupancy"]
        avg_sp = ld["avg_speed"]/max_speed
        r = lane_risks[lane]/2.0
        lane_features += [occ, avg_sp, r]

    ego_x_norm = ego_x/2000.0
    current_lane_norm = current_lane_index/6.0
    target_lane_norm = target_lane/6.0

    state = [ego_x_norm, ego_speed, current_lane_norm, target_lane_norm] + lane_features
    rl_state = np.array(state, dtype=np.float32)

    logger.debug(f"RL State: position_norm={ego_x_norm}, speed_norm={ego_speed}, current_lane_norm={current_lane_norm}, target_lane_norm={target_lane_norm}, lane_features={lane_features}")
    return rl_state

def compute_reward(collision, stable, timeout, safety_violation, lane_data, lane_risks, acc, jerk, ego_state, target_lane, previous_distance_to_target, previous_lane_index):
    reward = 0.0
    reason = ""

    # Large reward for stable lane convergence
    if stable:
        reward += 100.0
        reason += "StableLaneChange "

    # Penalize collisions heavily
    if collision:
        reward -= 100.0
        reason += "Collision "

    # Penalize timeouts moderately
    if timeout:
        reward -= 50.0
        reason += "Timeout "

    # Penalize safety violations mildly
    if safety_violation:
        reward -= 10.0
        reason += "SafetyViolation "

    # Reduced jerk penalty
    if jerk is not None:
        jerk_penalty = abs(jerk)*0.001  # smaller penalty
        reward -= jerk_penalty

    # Reward for making progress towards the target lane
    current_lane_index = int(ego_state["lane_id"].split("_")[-1])
    distance_to_target = abs(current_lane_index - target_lane)

    # If we got closer to target lane this step
    if previous_lane_index is not None:
        old_distance = abs(previous_lane_index - target_lane)
        if distance_to_target < old_distance:
            # Moved one lane closer => +5 reward
            reward += 5.0
        elif distance_to_target > old_distance:
            # Moved away from target lane => -5 reward
            reward -= 5.0

    # Small positive reward each step without collision/timeouts/safety violation
    if not collision and not timeout and not safety_violation:
        reward += 0.2  # increased from 0.1 to give more frequent positive signal

    # Consider penalizing excessive speed if desired (assuming ego_speed in ego_state)
    ego_speed = ego_state["speed"]
    max_speed = 33.33
    if ego_speed > (max_speed * 0.8):
        # Slight penalty for going too fast if we want to encourage safer speeds
        reward -= 0.5
    else:
        # Slight positive if maintaining moderate speed
        reward += 0.1

    # Additional small positive reward for surviving a step without collision/timeout
    reward += 0.1

    return reward, reason.strip(), distance_to_target, current_lane_index

def run_episode(mpc, rl_agent, scenario_config, max_steps=3000):
    logger.info(f"Starting run_episode with scenario {scenario_config}.")
    traci.start(["sumo", "-c", scenario_config, "--start", "--no-step-log", "true"])

    WARMUP_STEPS = 500
    logger.info(f"Running warmup for {WARMUP_STEPS} steps to build traffic.")
    for _ in range(WARMUP_STEPS):
        traci.simulationStep()

    ego_info = add_ego_vehicle()
    ego_id = ego_info["id"]
    target_lane = ego_info["target_lane"]

    traci.simulationStep()
    initialize_logging()

    step = 0
    stable_lane_steps = 0
    done = False
    episode_reward = 0.0
    reason_for_termination = ""
    previous_distance_to_target = None
    previous_lane_index = None
    total_jerk = 0.0
    jerk_count = 0
    stable_lane_step = None

    # Let ego run in same lane for some steps:
    for _ in range(20):
        traci.simulationStep()
        step += 1
        if step>=max_steps:
            break

    while not done and step < max_steps:
        traci.simulationStep()
        step += 1

        if ego_id not in traci.vehicle.getIDList():
            logger.info("Ego vehicle no longer in simulation (possibly route ended).")
            done = True
            reason_for_termination = "EgoVehicleLeftSimulation"
            break

        lane_data = get_lane_info(ego_id)
        if lane_data is None:
            logger.warning("Lane data is None, ending episode.")
            done = True
            reason_for_termination = "NoLaneData"
            break

        ego_lane_id = traci.vehicle.getLaneID(ego_id)
        lane_risks = compute_lane_risk(lane_data, ego_lane_id)
        acc_val, jerk_val = compute_comfort_metrics(ego_id)

        collision = check_collision(ego_id)
        stable_lane_steps, stable = check_lane_change_completion(ego_id, target_lane, stable_lane_steps)
        if stable and stable_lane_step is None:
            stable_lane_step = step
        timeout = check_timeout(step)
        safety_violation = check_safety_violations(lane_data, ego_lane_id)

        ego_x, ego_y = traci.vehicle.getPosition(ego_id)
        ego_speed = traci.vehicle.getSpeed(ego_id)
        ego_state = {
            "position": (ego_x, ego_y),
            "speed": ego_speed,
            "lane_id": ego_lane_id,
            "target_lane": target_lane
        }

        rl_state = construct_rl_state(ego_state, lane_data, lane_risks)

        W_s, W_e, W_c = rl_agent.get_action(rl_state)
        mpc.set_weights(W_s, W_e, W_c)
        apply_mpc_weights(W_s, W_e, W_c)

        acc_cmd, lane_change_cmd = mpc.compute_control(ego_state, lane_data, lane_risks, target_lane)
        desired_speed = max(0, min(ego_speed + acc_cmd*0.1, mpc.max_speed))
        traci.vehicle.setSpeed(ego_id, desired_speed)

        if lane_change_cmd is not None:
            new_lane_index, duration = lane_change_cmd
            traci.vehicle.changeLane(ego_id, new_lane_index, int(duration / 0.1))

        reward, reason_segment, distance_to_target, current_lane_index = compute_reward(
            collision, stable, timeout, safety_violation, lane_data, lane_risks,
            acc_val, jerk_val, ego_state, target_lane, previous_distance_to_target, previous_lane_index
        )

        episode_reward += reward
        previous_distance_to_target = distance_to_target
        previous_lane_index = current_lane_index

        if jerk_val is not None:
            total_jerk += abs(jerk_val)
            jerk_count += 1

        if (collision or stable or timeout or safety_violation or step>=max_steps) and reason_for_termination=="":
            reason_for_termination = reason_segment if reason_segment!="" else "MaxStepsReached"

        done = collision or stable or timeout or safety_violation or step >= max_steps

        if ego_id in traci.vehicle.getIDList():
            next_ego_x, next_ego_y = traci.vehicle.getPosition(ego_id)
            next_ego_speed = traci.vehicle.getSpeed(ego_id)
            next_ego_state = {
                "position": (next_ego_x, next_ego_y),
                "speed": next_ego_speed,
                "lane_id": traci.vehicle.getLaneID(ego_id),
                "target_lane": target_lane
            }
            next_lane_data = get_lane_info(ego_id)
            next_lane_risks = compute_lane_risk(next_lane_data, next_ego_state["lane_id"]) if next_lane_data else lane_risks
            next_rl_state = construct_rl_state(next_ego_state, next_lane_data, next_lane_risks)
        else:
            next_rl_state = rl_state

        # Store transitions for all agents
        rl_agent.store_transition(rl_state, np.array([W_s,W_e,W_c],dtype=np.float32), reward, next_rl_state, done)

        # For PPO: do NOT update here. We'll update after the episode ends.
        # For SAC or TD3: they may update here. If needed:
        if hasattr(rl_agent, 'update_networks') and not isinstance(rl_agent, PPOAgent):
            # print("Updating for SAC and TD3")
            rl_agent.update_networks()

        log_data(step, ego_state, lane_risks, W_s, W_e, W_c, acc_val, jerk_val)

    traci.close()
    avg_jerk = (total_jerk / jerk_count) if jerk_count > 0 else 0.0
    print(f"Episode finished. Reward: {episode_reward:.2f}, Reason: {reason_for_termination}, AvgJerk: {avg_jerk:.4f}, StableLaneStep: {stable_lane_step}")

    # For PPO: update after episode
    if isinstance(rl_agent, PPOAgent):
        if len(rl_agent.rewards) > 0:
            rl_agent.update_networks_end_episode()
        else:
            # No data collected this episode, skip update
            pass

    return episode_reward, done, reason_for_termination, avg_jerk, stable_lane_step if stable_lane_step is not None else max_steps


if __name__ == "__main__":
    # Example usage if you just want to run a single episode with fixed agent (not learning)
    from mpc_controller import MPCController
    class DummyAgent:
        def get_action(self, state):
            return 1.0, 1.0, 1.0
        def store_transition(self,s,a,r,ns,d): pass
        def update_networks(self): pass

    scenario = random.choice(SCENARIO_CONFIG_FILES)
    mpc = MPCController()
    run_episode(mpc, DummyAgent(), scenario)
