import numpy as np
import matplotlib.pyplot as plt

g = 9.81
rho = 1.225
Cd = 0.47
A = 0.01
m = 0.045
r = 0.056
delta_t = 0.01

targets = [(10, 1), (8, 3)]
x_targets = [target[0] for target in targets]
y_targets = [target[1] for target in targets]

def simulate_trajectory(v_0, theta, x_0=0, y_0=0, targets=None):
    theta_rad = np.radians(theta)
    v_0x = v_0 * np.cos(theta_rad)
    v_0y = v_0 * np.sin(theta_rad)

    x = x_0
    y = y_0
    vx = v_0x
    vy = v_0y
    t = 0

    trajectory_x = [x]
    trajectory_y = [y]

    def compute_velocity(vx, vy):
        return np.sqrt(vx ** 2 + vy ** 2)

    while y >= 0:
        v = compute_velocity(vx, vy)

        F_drag_x = 0.5 * Cd * rho * A * v * vx
        F_drag_y = 0.5 * Cd * rho * A * v * vy

        ax = -F_drag_x / m
        ay = -g - F_drag_y / m

        vx += ax * delta_t
        vy += ay * delta_t

        x += vx * delta_t
        y += vy * delta_t

        trajectory_x.append(x)
        trajectory_y.append(y)

        if targets:
            for target in targets:
                if np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2) < r:
                    return True, trajectory_x, trajectory_y, target

    return False, trajectory_x, trajectory_y, None


def shooting_method(targets):
    angle = 45
    velocity = 12

    trajectories_x = []
    trajectories_y = []
    hit_targets = []
    attempts = 0
    max_attempts = 100

    while len(hit_targets) < len(targets) and attempts < max_attempts:
        success = False
        trajectory_x, trajectory_y = [], []

        success, trajectory_x, trajectory_y, hit_target = simulate_trajectory(velocity, angle, targets=targets)

        if success:
            hit_targets.append(hit_target)
            print(f"Ball hit the target at {hit_target} with angle {angle}° and velocity {velocity} m/s!")

        trajectories_x.append(trajectory_x)
        trajectories_y.append(trajectory_y)

        if len(hit_targets) < len(targets):
            angle += 1
            velocity += 0.5

        attempts += 1

    if len(hit_targets) == len(targets):
        print(f"All targets hit after {attempts} attempts!")
    else:
        print(f"Reached maximum attempts ({max_attempts}) without hitting all targets.")

    return trajectories_x, trajectories_y


all_trajectories_x, all_trajectories_y = shooting_method(targets)

plt.figure(figsize=(10, 6))

for trajectory_x, trajectory_y in zip(all_trajectories_x, all_trajectories_y):
    plt.plot(trajectory_x, trajectory_y, label="Ball Trajectory", alpha=0.5)

for target_x, target_y in zip(x_targets, y_targets):
    plt.scatter(target_x, target_y, color='red', label="Target Ball", zorder=5)

plt.title("Projectile Motion with Air Resistance and Multiple Targets")
plt.xlabel("Horizontal Position (m)")
plt.ylabel("Vertical Position (m)")
plt.legend()
plt.grid(True)

plt.xlim(0, 15)
plt.ylim(0, 15)

plt.savefig('xyz.png')
