import sys
import pdb
import numpy as np

import matplotlib.pyplot as plt
import pickle

# from mayavi import mlab

from utils import (
    load_points_from_csv,
    extract_available_regions,
    assign_initial_labeling,
    extract_leaps,
    calc_leap_threshold,
    extract_consistent_trajectories,
)


def main(file_name):

    file_path = f"./walk_dataset/measurement_20240604/{file_name}"

    time_points_dict = load_points_from_csv(file_path)
    timestamps = sorted(list(time_points_dict.keys()))
    point_nums = [len(time_points_dict[timestamp]) for timestamp in timestamps]

    available_regions = extract_available_regions(point_nums, 21)
    region_lengths = [len(region) for region in available_regions]
    focused_region = available_regions[np.argmax(region_lengths)]
    focused_timestamps = [timestamps[i] for i in focused_region]

    focused_time_points_dict = {
        timestamp: time_points_dict[timestamp] for timestamp in focused_timestamps
    }

    # point_nums = [len(focused_time_points_dict[timestamp]) for timestamp in focused_timestamps]
    

    consistent_trajectories = assign_initial_labeling(focused_time_points_dict, 0.03)
    # for label, trajectory in trajectories.items():
    #     leaps = extract_leaps(trajectory)
    #     plt.ylim(0, 0.05)
    #     plt.plot(leaps)
    # plt.show()

    # leaps = extract_all_leaps(trajectories)
    # leap_threshold = calc_leap_threshold(leaps)
    # leap_threshold = 0.03

    # consistent_trajectories = extract_consistent_trajectories(
    #     focused_time_points_dict, leap_threshold
    # )

    filtered_trajectories = {}
    for label, trajectory in consistent_trajectories.items():
        if len(trajectory) > 480:
            filtered_trajectories[label] = trajectory

    for label, trajectory in filtered_trajectories.items():
        times = np.array([point["time"] for point in trajectory])
        plt.scatter(times, [label] * len(times), color="black", s=5)
    plt.show()

    time_points_dict = {}
    for label, trajectory in filtered_trajectories.items():
        for point in trajectory:
            timestamp = round(point["time"], 5)
            if timestamp not in time_points_dict:
                time_points_dict[timestamp] = [
                    {"label": label, "point": point["point"]}
                ]
            else:
                time_points_dict[timestamp].append(
                    {"label": label, "point": point["point"]}
                )

    timestamps = sorted(list(time_points_dict.keys()))

    outfile = file_name
    outfile = outfile.replace(".csv", ".pickle")

    with open(outfile, "wb") as f:
        pickle.dump(time_points_dict, f)


if __name__ == "__main__":
    file_names = (
        "calib_180_pre.csv",
        "none_2_20240604.csv",
        "tanka_20240604.csv",
        "cb_tanka_20240604.csv",
        "cb_tanka_2_20240604.csv",
    )
    for file_name in file_names:
        main(file_name)
