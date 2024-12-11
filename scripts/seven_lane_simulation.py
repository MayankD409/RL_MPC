import os
import sys
import math
import random
import csv
import traci
import time
import logging

# Import the MPC controller
from mpc_controller import MPCController

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mpc_simulation.log"),  # Log to file
        logging.StreamHandler(sys.stdout)  # Log to console
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
    "edge_start_end_3": 20.0,   # slow lane
    "edge_start_end_4": 28.0,   # medium speed lane
    "edge_start_end_5": 30.0,  # fast lane
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
    # initial_lane = random.randint(0, 6)
    initial_lane = 0
    initial_speed = random.uniform(0, 10)
    # target_lane = random.choice([l for l in range(7) if l != initial_lane])
    target_lane = 5
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
            with open(LOG_FILE, "w", newline="") as f:
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

def run_simulation():
    # Number of steps to run before adding ego vehicle
    WARMUP_STEPS = 2000
    try:
        randomize_scenario()
        traci.start(["sumo-gui", "-c", SELECTED_CONFIG, "--start", "--no-step-log", "true"])
        logger.info("SUMO simulation started.")
    except Exception as e:
        logger.critical(f"Failed to start SUMO simulation: {e}")
        sys.exit(1)
    try:
        # Set distinct lane speeds
        set_lane_speeds()

        for _ in range(WARMUP_STEPS):
            traci.simulationStep()


        # Add ego vehicle
        ego_info = add_ego_vehicle()
        ego_id = ego_info["id"]
        target_lane = ego_info["target_lane"]

        traci.vehicle.setLaneChangeMode(ego_id, 1621)  # Set lane change mode to "strategic"

        traci.simulationStep()
        logger.info("Initial simulation step completed.")

        initialize_logging()

        step = 0
        stable_lane_steps = 0

        # Initialize MPC Controller
        mpc = MPCController(wheelbase=2.5, max_acc=2.0, max_decel=-4.5, max_speed=33.33, lane_width=3.2, horizon=10, dt=0.1)
        logger.info("MPC Controller initialized.")

        # Fixed weights for now (Person C baseline)
        W_s, W_e, W_c = 1.0, 1.0, 1.0
        mpc.set_weights(W_s, W_e, W_c)
        apply_mpc_weights(W_s, W_e, W_c)

        logger.info(f"Initial MPC weights set: W_s={W_s}, W_e={W_e}, W_c={W_c}")

        while True:
            traci.simulationStep()
            step += 1
            logger.debug(f"Simulation step {step} started.")


            vehicle_ids = traci.vehicle.getIDList()
            if ego_id not in vehicle_ids:
                logger.error(f"Vehicle '{ego_id}' has left the simulation (possibly route ended).")
                break

            lane_data = get_lane_info(ego_id)
            if lane_data is None:
                logger.error(f"Failed to retrieve lane data for '{ego_id}'. Ending episode.")
                break

            ego_lane_id = traci.vehicle.getLaneID(ego_id)
            lane_risks = compute_lane_risk(lane_data, ego_lane_id)

            # Compute comfort metrics
            acc, jerk = compute_comfort_metrics(ego_id)

            # Check collisions
            if check_collision(ego_id):
                logger.error("Collision occurred. Ending episode.")
                break

            # Check lane change completion
            stable_lane_steps, stable = check_lane_change_completion(ego_id, target_lane, stable_lane_steps)
            if stable:
                logger.info("Lane change completed and stabilized (success).")
                break

            # Check timeout
            if check_timeout(step):
                logger.warning("Timeout reached. Ending episode.")
                break

            # Check safety violations
            if check_safety_violations(lane_data, ego_lane_id):
                logger.warning("Safety violation detected. Ending episode.")
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

            logger.debug(f"Ego State at step {step}: {ego_state}")

            # Solve MPC
            acc_cmd, lane_change_cmd = mpc.compute_control(ego_state, lane_data, lane_risks, target_lane)
            logger.debug(f"MPC Control Commands at step {step}: Acceleration={acc_cmd}, Lane Change Command={lane_change_cmd}")

            # Apply controls
            try:
                desired_speed = max(0, min(ego_speed + acc_cmd*0.1, mpc.max_speed))
                traci.vehicle.setSpeed(ego_id, desired_speed)
                logger.debug(f"Applied desired_speed {desired_speed} to ego vehicle '{ego_id}'.")

                if lane_change_cmd is not None:
                    new_lane_index, duration = lane_change_cmd
                    traci.vehicle.changeLane(ego_id, new_lane_index, int(duration / mpc.dt))
                    logger.info(f"Lane change command issued: Change to lane {new_lane_index} over {duration} seconds.")
            except traci.exceptions.TraCIException as e:
                logger.error(f"Error applying control commands to ego vehicle '{ego_id}': {e}")

            # Log data for analysis
            log_data(step, ego_state, lane_risks, W_s, W_e, W_c, acc, jerk)

            delay = 0.1
            time.sleep(delay)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the simulation: {e}")
    finally:
        traci.close()
        logger.info("SUMO simulation closed.")
        print("Episode ended.")

if __name__ == "__main__":
    run_simulation()
