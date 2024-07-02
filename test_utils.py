import pytest
import numpy as np
from utils import estimate_next_position, load_points_from_csv

def test_estimate_next_position():

    csv_file = "./walk_dataset/measurement_20240604/cb_tanka_20240604.csv"

    time_points_dict = load_points_from_csv(csv_file)
    times = np.array(sorted(list(time_points_dict.keys())))
    frame_counts = np.array([len(time_points_dict[time]) for time in times])

    max_traced_frames = np.where(frame_counts == 23)[0]
    max_traced_times = np.array(times[max_traced_frames])

    initial_time = max_traced_times[0]
    initial_points = time_points_dict[initial_time]

    point_buffer = []
    time_buffer = []
    for initial_point in initial_points:
        trajectory = [initial_point]
        focused_point = initial_point
        for i in range(1, len(max_traced_times)):
            next_time = max_traced_times[i]
            next_points = time_points_dict[next_time]

            point_buffer.append(focused_point)
            time_buffer.append(next_time)

            min_dist = None
            if len(point_buffer) < 6:
                dists = np.array([np.linalg.norm(next_point - focused_point) for next_point in next_points])
                next_point = next_points[np.argmin(dists)]
                min_dist = np.min(dists)
            else:
                estimated_next_position = estimate_next_position(point_buffer)
                estimated_dists = np.array([np.linalg.norm(next_point - estimated_next_position) for next_point in next_points])
                min_dist = np.min(estimated_dists)
                next_point = next_points[np.argmin(estimated_dists)]

            assert min_dist < 0.01

            trajectory.append({'time': next_time, 'point': next_point})
            focused_point = next_point

        point_buffer.append(trajectory)

if __name__ == "__main__":
    pytest.main()
