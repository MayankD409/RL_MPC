import traci
import os
import sys

# Add SUMO_HOME to PATH
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def start_simulation():
    sumo_binary = "sumo-gui"  # Use "sumo" for headless mode
    sumo_cmd = [sumo_binary, "-c", "config/sumoconfig.sumo.cfg", "--collision.action", "warn"]
    traci.start(sumo_cmd)

def run_simulation():
    ego_id = "ego_flow.0"  # The ID of the ego vehicle
    collision_count = 0
    max_collisions = 5  # Maximum number of collisions before ending the simulation

    while collision_count < max_collisions:
        start_simulation()
        
        step = 0
        while step < 3600:  # Run for 3600 steps (1 hour if step length is 1 second)
            traci.simulationStep()

            # Check for collisions involving the ego vehicle
            if traci.simulation.getCollidingVehiclesNumber() > 0:
                colliding_vehicles = traci.simulation.getCollidingVehiclesIDList()
                if ego_id in colliding_vehicles:
                    print(f"Warning: Ego vehicle {ego_id} involved in collision at step {step}!")
                    collision_count += 1
                    break  # Exit the inner loop to restart the simulation

            # Ensure the ego vehicle stays in the simulation
            if ego_id not in traci.vehicle.getIDList():
                print(f"Ego vehicle {ego_id} not found. Restarting simulation.")
                break

            step += 1

        traci.close()
        print(f"Simulation ended. Restarting... (Collision count: {collision_count})")

    print(f"Simulation ended after {collision_count} collisions.")

if __name__ == "__main__":
    run_simulation()