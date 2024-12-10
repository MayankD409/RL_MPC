import os
import sys
import math
import random
import csv
import traci

# Path to the SUMO configuration file
# Person A can prepare multiple scenario config files or route files and select one randomly:
SCENARIO_CONFIG_FILES = [
    "config/sumocfg/mySimulation_light.sumocfg",
    "config/sumocfg/mySimulation_medium.sumocfg",
    "config/sumocfg/mySimulation_heavy.sumocfg"
]

# If these scenario files differ by different route files or traffic flows,
# Person A can maintain them. For demonstration, we assume these files exist.
# If not, Person A must create them with different route distributions.

# Set a random scenario for diversity
SELECTED_CONFIG = random.choice(SCENARIO_CONFIG_FILES)
# SELECTED_CONFIG = "config/sumocfg/mySimulation.sumocfg"

# -------------------------------
# Configuration Parameters
# -------------------------------
MAX_STEPS = 3000          # ~300s if step-length=0.1s
STABLE_STEPS_REQUIRED = 50
PROXIMITY_DISTANCE = 50.0
COLLISION_DISTANCE = 1.0
RISK_OCCUPANCY_HIGH = 0.4
RISK_OCCUPANCY_MED = 0.2
RISK_SPEED_LOW = 10.0
RISK_CLOSE_VEHICLE = 10.0

# Lane speed variations (for demonstration)
LANE_SPEEDS = {
    "edge_start_end_0": 25.0,   # slow lane
    "edge_start_end_1": 30.0,   # medium speed lane
    "edge_start_end_2": 33.33,  # fast lane
    "edge_end_start_0": 25.0,
    "edge_end_start_1": 30.0,
    "edge_end_start_2": 33.33
}

# Logging options
LOG_DATA = True
LOG_FILE = "simulation_data_log.csv"

# Track previous ego states for comfort metrics (jerk calculation)
previous_ego_speed = None
previous_ego_acc = None
previous_ego_time = None

# Track how long we've been in unsafe proximity
unsafe_proximity_count = 0
UNSAFE_PROXIMITY_THRESHOLD = 5  # Steps of continuous unsafe proximity before terminating

def apply_mpc_weights(W_s, W_e, W_c):
    """
    Placeholder function for applying MPC weights.
    Person B would call this after deciding on an action.
    Currently, just prints the weights to demonstrate the interface.
    """
    print(f"Applying MPC weights: Safety={W_s:.2f}, Efficiency={W_e:.2f}, Comfort={W_c:.2f}")

def add_ego_vehicle():
    """Add the ego vehicle with random initial conditions."""
    ego_id = "ego"
    initial_lane = random.randint(0, 2)
    initial_speed = random.uniform(0, 10)
    target_lane = random.choice([l for l in [0,1,2] if l != initial_lane])

    # Ensure route exists for ego
    if "ego_route" not in traci.route.getIDList():
        traci.route.add("ego_route", ["edge_start_end", "edge_end_start"])

    traci.vehicle.add(
        vehID=ego_id,
        routeID="ego_route",
        typeID="car",
        depart="0",
        departLane=str(initial_lane),
        departSpeed=str(initial_speed),
        departPos="0"
    )

    return {
        "id": ego_id,
        "initial_lane": initial_lane,
        "target_lane": target_lane,
        "initial_speed": initial_speed
    }

def set_lane_speeds():
    """Set distinct lane speeds to differentiate lane attributes."""
    for lane_id, speed in LANE_SPEEDS.items():
        try:
            traci.lane.setMaxSpeed(lane_id, speed)
        except traci.exceptions.TraCIException:
            pass

def get_lane_info(ego_id):
    """Retrieve lane occupancy, avg speed, and proximity data relative to ego."""
    try:
        ego_lane_id = traci.vehicle.getLaneID(ego_id)
        ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    except traci.exceptions.TraCIException:
        print(f"Vehicle '{ego_id}' not found in the simulation.")
        return None

    lanes = [f"edge_start_end_{i}" for i in range(3)] + [f"edge_end_start_{i}" for i in range(3)]
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
                except traci.exceptions.TraCIException:
                    continue

        lane_data[lane] = {
            "occupancy": occupancy,
            "avg_speed": avg_speed,
            "proximity_count": proximity_count,
            "proximity_min_dist": min(proximity_dists) if proximity_dists else None
        }

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

    return lane_risks

def check_collision(ego_id):
    """Check if a collision occurred based on proximity."""
    try:
        ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    except traci.exceptions.TraCIException:
        return True

    vehicles = traci.vehicle.getIDList()
    for v in vehicles:
        if v != ego_id:
            try:
                x, y = traci.vehicle.getPosition(v)
                dist = math.sqrt((x - ego_x)**2 + (y - ego_y)**2)
                if dist < COLLISION_DISTANCE:
                    return True
            except traci.exceptions.TraCIException:
                continue
    return False

def check_lane_change_completion(ego_id, target_lane, stable_count):
    """Check if ego has stabilized in target lane."""
    try:
        current_lane_id = traci.vehicle.getLaneID(ego_id)
        current_lane_index = int(current_lane_id.split("_")[-1])
    except traci.exceptions.TraCIException:
        return 0, False

    if current_lane_index == target_lane:
        stable_count += 1
    else:
        stable_count = 0

    stable = stable_count >= STABLE_STEPS_REQUIRED
    return stable_count, stable

def check_timeout(step):
    """Check if episode exceeded max steps."""
    return step >= MAX_STEPS

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
    else:
        unsafe_proximity_count = 0

    # If we exceed the threshold, end episode
    if unsafe_proximity_count >= UNSAFE_PROXIMITY_THRESHOLD:
        return True
    return False

def compute_comfort_metrics(ego_id, step_time=0.1):
    """
    Compute approximate jerk as a comfort metric.
    SUMO does not directly provide acceleration or jerk, but we can estimate acceleration:
    acc = (v_current - v_previous)/delta_t
    jerk = (acc_current - acc_previous)/delta_t
    """
    global previous_ego_speed, previous_ego_acc, previous_ego_time

    try:
        ego_speed = traci.vehicle.getSpeed(ego_id)
    except:
        return None, None

    # Compute acceleration if previous speed is available
    if previous_ego_speed is not None:
        acc = (ego_speed - previous_ego_speed) / step_time
    else:
        acc = 0.0

    # Compute jerk if previous acceleration is available
    if previous_ego_acc is not None:
        jerk = (acc - previous_ego_acc) / step_time
    else:
        jerk = 0.0

    # Update previous states
    previous_ego_speed = ego_speed
    previous_ego_acc = acc

    return acc, jerk

def randomize_scenario():
    """
    Randomize scenario by choosing a different config file and possibly different routes.
    Since SUMO configuration is started externally, we just selected SELECTED_CONFIG.
    Person A can create multiple scenario config files with different route distributions.
    """
    print(f"Selected scenario config: {SELECTED_CONFIG}")

def initialize_logging():
    if LOG_DATA:
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "ego_x", "ego_y", "ego_speed", "ego_lane", 
                             "lane_risks", "W_s", "W_e", "W_c", "acc", "jerk"])

def log_data(step, ego_state, lane_risks, W_s, W_e, W_c, acc, jerk):
    if LOG_DATA:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, ego_state["position"][0], ego_state["position"][1],
                             ego_state["speed"], ego_state["lane_id"], lane_risks,
                             W_s, W_e, W_c, acc, jerk])

def run_simulation():
    randomize_scenario()
    traci.start(["sumo-gui", "-c", SELECTED_CONFIG, "--start", "--no-step-log", "true"])

    # Set distinct lane speeds
    set_lane_speeds()

    # Add ego vehicle
    ego_info = add_ego_vehicle()
    ego_id = ego_info["id"]
    target_lane = ego_info["target_lane"]

    traci.simulationStep()

    initialize_logging()

    step = 0
    stable_lane_steps = 0

    # Initial guess for MPC weights (Person B would adjust these)
    W_s, W_e, W_c = 1.0, 1.0, 1.0
    apply_mpc_weights(W_s, W_e, W_c)

    while True:
        traci.simulationStep()
        step += 1

        vehicle_ids = traci.vehicle.getIDList()
        if ego_id not in vehicle_ids:
            print(f"Vehicle '{ego_id}' has left the simulation (possibly route ended).")
            break

        lane_data = get_lane_info(ego_id)
        if lane_data is None:
            print(f"Failed to retrieve lane data for '{ego_id}'. Ending episode.")
            break

        ego_lane_id = traci.vehicle.getLaneID(ego_id)
        lane_risks = compute_lane_risk(lane_data, ego_lane_id)

        # Compute comfort metrics
        acc, jerk = compute_comfort_metrics(ego_id)

        # Check collisions
        if check_collision(ego_id):
            print("Collision occurred. Ending episode.")
            break

        # Check lane change completion
        stable_lane_steps, stable = check_lane_change_completion(ego_id, target_lane, stable_lane_steps)
        if stable:
            print("Lane change completed and stabilized (success).")
            break

        # Check timeout
        if check_timeout(step):
            print("Timeout reached. Ending episode.")
            break

        # Check safety violations
        if check_safety_violations(lane_data, ego_lane_id):
            print("Safety violation detected. Ending episode.")
            break

        # Gather ego state for RL
        ego_x, ego_y = traci.vehicle.getPosition(ego_id)
        ego_speed = traci.vehicle.getSpeed(ego_id)
        ego_state = {
            "position": (ego_x, ego_y),
            "speed": ego_speed,
            "lane_id": ego_lane_id,
            "target_lane": target_lane
        }

        # Example: Person B might send updated weights based on their RL policy:
        # For now, we keep them constant or just print them
        # apply_mpc_weights(W_s, W_e, W_c)

        # Log data for analysis
        log_data(step, ego_state, lane_risks, W_s, W_e, W_c, acc, jerk)

        # In a real integration:
        # Person B would read state_data, compute action (W_s, W_e, W_c), call apply_mpc_weights,
        # and then Person A's code continues to next step.

    traci.close()
    print("Episode ended.")

if __name__ == "__main__":
    run_simulation()
