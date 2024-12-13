import math
import cvxpy as cp

class MPCController:
    def __init__(self, wheelbase=2.5, max_acc=2.0, max_decel=-4.5, max_speed=33.33, lane_width=3.2, horizon=10, dt=0.1):
        """
        Initialize MPC parameters.
        """
        self.L = wheelbase
        self.max_acc = max_acc
        self.max_decel = max_decel
        self.max_speed = max_speed
        self.lane_width = lane_width
        self.horizon = horizon
        self.dt = dt

        # Default weights
        self.W_s = 1.0
        self.W_e = 1.0
        self.W_c = 1.0

    def set_weights(self, W_s, W_e, W_c):
        """Update the MPC cost weights based on RL or fixed baseline."""
        self.W_s = W_s
        self.W_e = W_e
        self.W_c = W_c

    def compute_control(self, ego_state, lane_data, lane_risks, target_lane):
        """
        Compute the optimal control inputs (acc and lane-change command) using MPC and incremental lane changes.
        """
        ego_x, ego_y = ego_state["position"]
        ego_v = ego_state["speed"]
        current_lane_id = ego_state["lane_id"]
        current_lane_index = int(current_lane_id.split("_")[-1])

        # Determine desired speed based on current lane risk
        current_lane_risk = lane_risks.get(current_lane_id, 1)
        if current_lane_risk == 2:
            desired_speed = min(self.max_speed, ego_v)
        elif current_lane_risk == 1:
            desired_speed = min(self.max_speed, ego_v + 5.0)
        else:
            desired_speed = self.max_speed

        # If lane is crowded, reduce acceleration desire
        proximity_count = lane_data[current_lane_id]["proximity_count"]
        if proximity_count > 3:
            desired_speed = min(desired_speed, ego_v)

        # Solve MPC for longitudinal acceleration
        v = cp.Variable(self.horizon+1)
        a = cp.Variable(self.horizon)

        constraints = [v[0] == ego_v]
        for k in range(self.horizon):
            constraints += [v[k+1] == v[k] + a[k]*self.dt]

        for k in range(self.horizon):
            constraints += [a[k] <= self.max_acc, a[k] >= self.max_decel]
        for k in range(self.horizon+1):
            constraints += [v[k] >= 0, v[k] <= self.max_speed]

        if math.isnan(desired_speed) or math.isinf(desired_speed):
            desired_speed = self.max_speed  # fallback

        cost = 0
        for k in range(self.horizon+1):
            cost += self.W_e * cp.square(v[k] - desired_speed)
        for k in range(self.horizon):
            cost += self.W_c * cp.square(a[k])
        for k in range(self.horizon-1):
            cost += self.W_c * cp.square(a[k+1] - a[k])

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            acc_cmd = 0.0
        else:
            acc_cmd = a.value[0]

        # Incremental lane change logic
        lane_change_cmd = None
        if target_lane != current_lane_index:
            # Determine direction of lane change
            direction = 1 if target_lane > current_lane_index else -1
            next_lane_index = current_lane_index + direction
            next_lane_id = current_lane_id.rsplit("_", 1)[0] + f"_{next_lane_index}"

            # Check lane risk of the adjacent lane
            next_lane_risk = lane_risks.get(next_lane_id, 1)
            if next_lane_risk <= current_lane_risk:
                lane_change_cmd = (next_lane_index, 2.0)

        return acc_cmd, lane_change_cmd
