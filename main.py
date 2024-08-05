import polars as pl
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_points_from_csv,
    extract_available_regions,
    assign_initial_labeling,
    extract_all_leaps,
)


def main(file_path):

    time_points_dict = load_points_from_csv(file_path)
    timestamps = sorted(list(time_points_dict.keys()))
    point_nums = [len(time_points_dict[timestamp]) for timestamp in timestamps]

    available_regions = extract_available_regions(point_nums, 21)
    region_lengths = [len(region) for region in available_regions]
    focused_region = available_regions[np.argmax(region_lengths)]
    focused_timestamps = [timestamps[i] for i in focused_region]

    trajectories = assign_initial_labeling(focused_timestamps, time_points_dict)
    leaps = extract_all_leaps(trajectories)

    plt.hist(leaps, bins=50, edgecolor="black")
    plt.title("Histogram of Leaps")
    plt.xlabel("Leap Distance")
    plt.ylabel("Frequency")
    plt.show()

    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv[1])
