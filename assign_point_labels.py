import pdb
import sys
import os
from utils import load_points_from_csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm


def extract_labeled_trajectories(filepath):

    time_points_dict = load_points_from_csv(filepath)

    # filter out trajectories with less than 21 points
    valid_frames_dict = {
        time: value for time, value in time_points_dict.items() if len(value) >= 21
    }
    valid_frames = sorted(list(valid_frames_dict.keys()))

    # extract successive frames in valid_frames_dict
    successive_valid_frames = []
    successive_frames = []
    for i in range(1, len(valid_frames)):
        current = valid_frames[i]
        last = valid_frames[i - 1]
        if current - last < 1 / 100:
            successive_frames.append(current)
        else:
            if len(successive_frames) > 0:
                successive_valid_frames.append(successive_frames)
            successive_frames = []
    if len(successive_frames) > 0:
        successive_valid_frames.append(successive_frames)

    successive_lengths = np.array([len(frame) for frame in successive_valid_frames])
    longest_idx = np.argmax(successive_lengths)

    focused_frames = successive_valid_frames[longest_idx]

    focused_time_points = {
        focused: valid_frames_dict[focused] for focused in focused_frames
    }

    num_points = np.array([len(focused_time_points[frame]) for frame in focused_frames])
    start_indices = np.where(num_points == 21)[0]
    if len(start_indices) == 0:
        start_idx = 0
    else:
        start_idx = start_indices[0]
    focused_frames = focused_frames[start_idx:]

    labeled_trajectories = {}
    label = 0
    for i, frame in enumerate(focused_frames):
        points = focused_time_points[frame]
        origin = np.array([0, 0, 0])
        sorted_points = sorted(points, key=lambda p: np.linalg.norm(p - origin))
        points = sorted_points

        if i == 0:
            # Sort points based on their distance from the origin
            for point in points:
                labeled_trajectories[label] = [point]
                label += 1
        else:
            for label, trajectory in labeled_trajectories.items():
                # estimated = estimate_next_position(trajectory)
                # tmp = [
                #     np.linalg.norm(trajectory[i + 1] - trajectory[i])
                #     for i in range(len(trajectory) - 1)
                # ]
                tail = trajectory[-1]
                dists = [np.linalg.norm(point - tail) for point in points]
                closest_idx = np.argmin(dists)
                # if dists[closest_idx] > 0.01:
                #     pdb.set_trace()

                trajectory.append(points[closest_idx])
                points = np.delete(points, closest_idx, axis=0)

    return labeled_trajectories


if __name__ == "__main__":

    # Get the list of CSV files in the walk_dataset/measurement_20240604 directory
    dataset_dir = "walk_dataset/measurement_20240604"
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

    for csv_file in tqdm.tqdm(csv_files):
        filepath = "./" + dataset_dir + "/" + csv_file
        labeled_trajectories = extract_labeled_trajectories(filepath)

        leaping_positions = []
        for label, trajectory in labeled_trajectories.items():
            diffs = []
            for i in range(1, len(trajectory)):
                diff = np.linalg.norm(trajectory[i] - trajectory[i - 1])
                diffs.append(diff)
            diffs = np.array(diffs)
            leapings = np.where(diffs > 0.1)[0]
            leaping_positions += leapings.tolist()

        leaping_positions = sorted(leaping_positions)
        if len(leaping_positions) > 0:
            long_stable_idx = np.argmax(np.diff(leaping_positions))
            start = leaping_positions[long_stable_idx] + 1
            end = leaping_positions[long_stable_idx + 1]
        else:
            start = 0
            end = -1

        shorten_trajectories = {}
        for label, trajectory in labeled_trajectories.items():
            shorten_trajectories[label] = trajectory[start:end]

        output_path = csv_file.replace(".csv", "_trajectories.pickle")
        with open(output_path, "wb") as f:
            pickle.dump(shorten_trajectories, f)
