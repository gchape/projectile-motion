import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81
rho = 1.225
Cd = 0.47
A = 0.01
m = 0.045
r = 0.15
delta_t = 0.01


def apply_sobel_edge_detection(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    rows, cols = gray_image.shape

    grad_x = np.zeros_like(gray_image, dtype=np.float32)
    grad_y = np.zeros_like(gray_image, dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            roi = gray_image[i - 1:i + 2, j - 1:j + 2]
            grad_x[i, j] = np.sum(roi * sobel_x)
            grad_y[i, j] = np.sum(roi * sobel_y)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = np.uint8(np.clip(grad_magnitude, 0, 255))

    _, edges = cv2.threshold(grad_magnitude, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                targets.append((cx, cy))

    return targets


def simulate_trajectory(v_0, theta, x_0=2, y_0=2, targets=None):
    theta_rad = np.radians(theta)
    v_0x = v_0 * np.cos(theta_rad)
    v_0y = v_0 * np.sin(theta_rad)

    x = x_0
    y = y_0
    vx = v_0x
    vy = v_0y

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
    angle = 13
    velocity = 20

    trajectories_x = []
    trajectories_y = []
    hit_targets = []
    attempts = 0
    max_attempts = 100

    remaining_targets = targets.copy()

    while len(hit_targets) < len(targets) and attempts < max_attempts:
        success, x, y, hit_target = simulate_trajectory(velocity, angle, targets=remaining_targets)

        if success:
            hit_targets.append(hit_target)
            remaining_targets.remove(hit_target)
            trajectories_x.append(x)
            trajectories_y.append(y)
            print(f"Ball hit the target at {hit_target} with angle {angle}° and velocity {velocity} m/s!")

        if len(hit_targets) < len(targets):
            angle += 1.5
            velocity += 0.5

        attempts += 1

    if len(hit_targets) == len(targets):
        print(f"All targets hit after {attempts} attempts!")
    else:
        print(f"Reached maximum attempts ({max_attempts}) without hitting all targets.")

    return trajectories_x, trajectories_y


def generate_trajectory_plot(image_path, x_0=2, y_0=2, plot_width=15, plot_height=15):
    image = cv2.imread(image_path)
    targets = apply_sobel_edge_detection(image_path)

    image_width = image.shape[1]
    image_height = image.shape[0]

    scaled_targets = []
    for target in targets:
        scaled_x = (target[0] / image_width) * plot_width
        scaled_y = (image_height - target[1]) / image_height * plot_height
        scaled_targets.append((scaled_x, scaled_y))

    all_trajectories_x, all_trajectories_y = shooting_method(scaled_targets)

    plt.figure(figsize=(10, 6))

    for target in scaled_targets:
        plt.scatter(target[0], target[1], color='red', label='Target', s=100)

    for trajectory_x, trajectory_y in zip(all_trajectories_x, all_trajectories_y):
        plt.plot(trajectory_x, trajectory_y, label="Ball Trajectory", alpha=0.5)

    plt.scatter(x_0, y_0, color='blue', s=50, label="Starting Point", zorder=6)

    plt.xlabel("Horizontal Position (m)")
    plt.ylabel("Vertical Position (m)")
    plt.grid(True)

    plt.xlim(0, plot_width)
    plt.ylim(0, plot_height)

    plt.savefig('trajectory_motion_plot.png')


def generate_trajectory_animation(image_path, x_0=2, y_0=2, plot_width=15, plot_height=15,
                                  video_filename="out/trajectory_motion.mp4", fps=30):
    image = cv2.imread(image_path)
    targets = apply_sobel_edge_detection(image_path)

    scaled_targets = [((target[0] / image.shape[1]) * plot_width,
                       (image.shape[0] - target[1]) / image.shape[0] * plot_height) for target in targets]

    all_trajectories_x, all_trajectories_y = shooting_method(scaled_targets)

    total_points = sum(len(trajectory) for trajectory in all_trajectories_x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, plot_width)
    ax.set_ylim(0, plot_height)
    ax.set_xlabel("Horizontal Position (m)")
    ax.set_ylabel("Vertical Position (m)")
    ax.grid(True)

    start_point = ax.scatter(x_0, y_0, color='blue', s=50, zorder=6)
    [ax.scatter(target[0], target[1], color='red', s=100) for target in scaled_targets]
    trajectory_line, = ax.plot([], [], alpha=0.5)

    def update_frame(frame):
        trajectory_index = 0
        point_index = frame

        while point_index >= len(all_trajectories_x[trajectory_index]):
            point_index -= len(all_trajectories_x[trajectory_index])
            trajectory_index += 1

        x_data = all_trajectories_x[trajectory_index][point_index]
        y_data = all_trajectories_y[trajectory_index][point_index]

        trajectory_line.set_data(
            all_trajectories_x[trajectory_index][:point_index + 1],
            all_trajectories_y[trajectory_index][:point_index + 1]
        )

        start_point.set_offsets([x_data, y_data])

        return trajectory_line, start_point

    ani = FuncAnimation(fig, update_frame, frames=total_points, repeat=False)
    ani.save(video_filename, writer='ffmpeg', fps=fps)

    plt.close(fig)


generate_trajectory_animation('img/balls2.png', x_0=2, y_0=2, plot_width=15, plot_height=15)
