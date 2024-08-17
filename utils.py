import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb


def get_nearest_trajectory(initial_point, timestamps, time_points_dict, min_dist):

    residual = time_points_dict.copy()

    focused_point = initial_point
    trajectory = [{"time": timestamps[0], "point": initial_point}]
    for i in range(1, len(timestamps)):
        next_points = residual[timestamps[i]]
        if len(next_points) == 0:
            loop_broken = True
            break
        dists = np.array(
            [np.linalg.norm(next_point - focused_point) for next_point in next_points]
        )

        if min_dist > 0:
            nearest_dist = np.min(dists)
            if nearest_dist > min_dist:
                break
            else:
                nearest_index = np.argmin(dists)
        else:
            dists = np.array(
                [np.linalg.norm(next_point - focused_point) for next_point in next_points]
            )
            nearest_index = np.argmin(dists)

        focused_point = next_points[nearest_index]
        trajectory.append({"time": timestamps[i], "point": focused_point})

        del residual[timestamps[i]][nearest_index]

    return trajectory, residual

def assign_initial_labeling(time_points_dict, min_dist=0.0):

    timestamps = sorted(list(time_points_dict.keys()))
    initial_points = time_points_dict[timestamps[0]]
    trajectories = {}
    for i, point in enumerate(initial_points):
        trajectories[i], time_points_dict = get_nearest_trajectory(
            point, timestamps, time_points_dict, min_dist
        )

    return trajectories


def extract_leaps(trajectory):

    leaps = []
    for i in range(1, len(trajectory)):
        distance = np.linalg.norm(trajectory[i] - trajectory[i - 1])
        leaps.append(distance)

    return leaps


def extract_all_leaps(trajectories):

    all_leaps = []
    for label, trajectory in trajectories.items():
        leaps = extract_leaps(trajectory)
        all_leaps += leaps

    return all_leaps


def check_point_identity(point1, point2):
    return np.allclose(point1, point2)


def calc_leap_threshold(leaps):
    # Detect outliers in leaps
    leaps_array = np.array(leaps)
    mean = np.mean(leaps_array)
    std_dev = np.std(leaps_array)
    threshold = 3  # Z-score threshold for outlier detection

    outliers = []
    for leap in leaps:
        z_score = (leap - mean) / std_dev
        if np.abs(z_score) > threshold:
            outliers.append(leap)

    threshold = np.min(outliers)
    print("Detected outlier threshold in leaps:", threshold)

    return threshold


def extract_not_labelled_points(points_pool):

    timestamps = sorted(list(points_pool.keys()))

    not_labelled_points = points_pool[timestamps[0]]
    del points_pool[timestamps[0]]

    return not_labelled_points, timestamps[0], points_pool


def extract_consistent_trajectories(time_points_dict, leap_threshold):

    points_pool = time_points_dict.copy()

    consistent_trajectories = {}
    label = 0
    while len(points_pool) > 0:
        not_labelled_points, timestamp, points_pool = extract_not_labelled_points(
            points_pool
        )
        timestamps = sorted(list(points_pool.keys()))
        for point in not_labelled_points:
            focused_point = point
            trajectory = [{"time": timestamp, "point": focused_point}]
            for timestamp in timestamps:
                next_points = points_pool[timestamp]
                if len(next_points) == 0:
                    break

                dists = np.array(
                    [
                        np.linalg.norm(next_point - focused_point)
                        for next_point in next_points
                    ]
                )
                if np.min(dists) >= leap_threshold:
                    break

                focused_point = next_points[np.argmin(dists)]
                trajectory.append({"time": timestamp, "point": focused_point})
                points_pool[timestamp] = np.delete(
                    points_pool[timestamp], np.argmin(dists), axis=0
                )
            if len(trajectory) > 1:
                consistent_trajectories[label] = trajectory
                label += 1

    return consistent_trajectories


def extract_available_regions(point_nums, available_threshold):

    available_regions = []
    consecutive_region = []
    for i in range(1, len(point_nums)):
        if (available_threshold <= point_nums[i]) and (
            available_threshold <= point_nums[i - 1]
        ):
            consecutive_region.append(i)
        else:
            if len(consecutive_region) > 0:
                available_regions.append(consecutive_region)
                consecutive_region = []

    if len(consecutive_region) > 0:
        available_regions.append(consecutive_region)

    return available_regions


def load_points_from_csv(csv_file):
    df = pd.read_csv(
        csv_file, comment="#"
    )  # Here you can add the code to process each CSV file

    time_points_dict = {}
    for index, row in df.iterrows():
        time = row["Time"]
        points = []
        for i in range(1, 24):  # There are 23 points as per the data
            point = (row[f"{i}(X)"], row[f"{i}(Y)"], row[f"{i}(Z)"])
            if "*" not in point:
                point = np.array(tuple(float(coord) for coord in point))
                points.append(point)
        time_points_dict[time] = points

    return time_points_dict


def estimate_next_position(point_buffer):

    if len(point_buffer) < 6:
        raise ValueError(
            "Not enough points in buffer to estimate next position using 5th order derivatives."
        )

    recent_points = np.array(point_buffer[-6:])

    first_derivatives = np.diff(recent_points, axis=0)
    second_derivatives = np.diff(first_derivatives, axis=0)
    third_derivatives = np.diff(second_derivatives, axis=0)
    # fourth_derivatives = np.diff(third_derivatives, axis=0)
    # fifth_derivatives = np.diff(fourth_derivatives, axis=0)

    next_position = (
        recent_points[-1]
        + first_derivatives[-1]
        + second_derivatives[-1] / 2
        + third_derivatives[-1] / 6
        # + fourth_derivatives[-1] / 24
        # + fifth_derivatives[-1] / 120
    )

    return next_position


def forecast_next_position(data_buffer):

    data_buffer = np.array(data_buffer)

    d = 3
    p = len(data_buffer)
    q = int(p / 2)

    arima_x = sm.tsa.SARIMAX(data_buffer[:, 0], order=(p, d, q)).fit(maxiter=1000)

    arima_y = sm.tsa.SARIMAX(data_buffer[:, 1], order=(p, d, q)).fit(maxiter=1000)

    arima_z = sm.tsa.SARIMAX(data_buffer[:, 2], order=(p, d, q)).fit(maxiter=1000)

    pred_x = arima_x.forecast(1)
    pred_y = arima_y.forecast(1)
    pred_z = arima_z.forecast(1)

    return np.array((pred_x, pred_y, pred_z))


def extract_trajectory_arima(initial_point, traced_times, time_points_dict):

    focused_point = initial_point
    initial_time = traced_times[0]
    trajectory = [{"time": initial_time, "point": focused_point}]
    point_buffer = [focused_point]

    for i in range(1, len(traced_times)):
        next_time = traced_times[i]
        next_points = time_points_dict[next_time]

        if 240 < len(point_buffer):
            forecasted_point = forecast_next_position(point_buffer)
            estimated_dists = np.array(
                [
                    np.linalg.norm(next_point - forecasted_point)
                    for next_point in next_points
                ]
            )
            next_point = next_points[np.argmin(estimated_dists)]

            trajectory.append({"time": next_time, "point": next_point})
            point_buffer.append(next_point)
            point_buffer.pop(0)
        else:
            dists = np.array(
                [
                    np.linalg.norm(next_point - focused_point)
                    for next_point in next_points
                ]
            )
            next_point = next_points[np.argmin(dists)]

            trajectory.append({"time": next_time, "point": next_point})
            point_buffer.append(next_point)

        focused_point = next_point

    pdb.set_trace()
    return trajectory


def extract_trajectory(initial_point, traced_times, time_points_dict):

    focused_point = initial_point
    initial_time = traced_times[0]
    trajectory = [{"time": initial_time, "point": focused_point}]

    time_buffer = []
    point_buffer = []

    min_dists = []
    for i in range(1, len(traced_times)):
        next_time = traced_times[i]
        next_points = time_points_dict[next_time]

        time_buffer.append(next_time)
        point_buffer.append(focused_point)

        """
        if 5 < len(point_buffer):
            estimated_next_position = estimate_next_position(point_buffer)
            dists = np.array([np.linalg.norm(next_point - estimated_next_position) for next_point in next_points])
            next_point = next_points[np.argmin(dists)]
        else:
            dists = np.array([np.linalg.norm(next_point - focused_point) for next_point in next_points])
            next_point = next_points[np.argmin(dists)]
        """

        dists = np.array(
            [np.linalg.norm(next_point - focused_point) for next_point in next_points]
        )
        next_point = next_points[np.argmin(dists)]
        min_dists.append(np.min(dists))

        trajectory.append({"time": next_time, "point": next_point})
        focused_point = next_point

        """
        dists = np.array([np.linalg.norm(next_point - focused_point) for next_point in next_points])
        next_point = next_points[np.argmin(dists)]
        """

    plt.plot(min_dists)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_vals = [point["point"][0] for point in trajectory]
    y_vals = [point["point"][1] for point in trajectory]
    z_vals = [point["point"][2] for point in trajectory]
    ax.scatter(
        x_vals, y_vals, z_vals, color=plt.cm.rainbow(np.linspace(0, 1, len(x_vals)))
    )
    # ax.plot(x_vals, y_vals, z_vals)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    plt.title("3D Trajectory Plot")
    plt.show()

    return trajectory


def extract_trajectory_reverse(initial_point, traced_times, time_points_dict):

    focused_point = initial_point
    initial_time = traced_times[0]
    trajectory = [{"time": initial_time, "point": focused_point}]

    for i in range(len(traced_times) - 1, 0, -1):
        next_time = traced_times[i]
        next_points = time_points_dict[next_time]

        dists = np.array(
            [np.linalg.norm(next_point - focused_point) for next_point in next_points]
        )
        next_point = next_points[np.argmin(dists)]

        trajectory.insert(0, {"time": next_time, "point": next_point})
        focused_point = next_point

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_vals = [point['point'][0] for point in trajectory]
    y_vals = [point['point'][1] for point in trajectory]
    z_vals = [point['point'][2] for point in trajectory]
    ax.scatter(x_vals, y_vals, z_vals, color=plt.cm.rainbow(np.linspace(0, 1, len(x_vals))))
    # ax.plot(x_vals, y_vals, z_vals)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    plt.title('3D Trajectory Plot')
    plt.show()
    """

    return trajectory


def linear_interpolate_trajectory(trajectory):

    times = [point["time"] for point in trajectory]
    skipped_frames = np.diff(times) / (1 / 120)
    skipped_frames = np.round(skipped_frames, 1)
    skipped_frames = skipped_frames.astype(int)

    # Utilize linear interpolation to fill in the gaps
    interpolated_trajectory = []
    for k in range(len(trajectory) - 1):
        if 1 < skipped_frames[k]:
            disp_diff = (
                trajectory[k + 1]["point"] - trajectory[k]["point"]
            ) / skipped_frames[k]
            time_diff = (
                trajectory[k + 1]["time"] - trajectory[k]["time"]
            ) / skipped_frames[k]
            interpolated_trajectory.append(trajectory[k])
            for l in range(1, skipped_frames[k]):
                interpolated_trajectory.append(
                    {
                        "time": trajectory[k]["time"] + l * time_diff,
                        "point": trajectory[k]["point"] + l * disp_diff,
                    }
                )
        else:
            interpolated_trajectory.append(trajectory[k])

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_vals = [point['point'][0] for point in interpolated_trajectory]
    y_vals = [point['point'][1] for point in interpolated_trajectory]
    z_vals = [point['point'][2] for point in interpolated_trajectory]
    ax.scatter(x_vals, y_vals, z_vals, color=plt.cm.rainbow(np.linspace(0, 1, len(x_vals))))

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    plt.title('3D Trajectory Plot')
    plt.show()
    """

    return interpolated_trajectory
