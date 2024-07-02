import pandas as pd
import pdb
import numpy as np
import sys
import tqdm
import pickle

import os
import glob

# Specify the directory containing the pickle files
pickle_directory_path = './'

# List all pickle files in the specified directory
pickle_files = glob.glob(os.path.join(pickle_directory_path, '*.pickle'))

# Loop through each pickle file and load it
for pickle_file in pickle_files:

    with open(pickle_file, 'rb') as file:
        time_points_dict = pickle.load(file)

    times = sorted(list(time_points_dict.keys()))

    initial_points = time_points_dict[times[0]]
    # Loop through each point in initial_points and process them

    label = 1
    trajectories = {}
    for point in initial_points:

        focused_point = point

        trajectories[label] = [{'time': times[0], 'point': focused_point}]
        for i in range(1, len(times)):
            next_points = time_points_dict[times[i]]

            min_distance = float('inf')
            closest_point = None
            for next_point in next_points:
                distance = np.linalg.norm(np.array(focused_point) - np.array(next_point))
                if distance < min_distance:
                    min_distance = distance
                    closest_point = next_point

            if 0.03 < min_distance:
                print(f'traced length: {i}/{len(times)}')
                break
            else:
                trajectories[label].append({'time': times[i], 'point': closest_point})

            focused_point = closest_point

        label += 1

    # Convert trajectories to a dictionary with time as key and list of points as values
    time_to_points_dict = {}
    for traj_label, traj_points in trajectories.items():
        for point_info in traj_points:
            time = point_info['time']
            point = point_info['point']
            if time not in time_to_points_dict:
                time_to_points_dict[time] = []
            time_to_points_dict[time].append({'label': traj_label, 'point': point})

    with open('time_points_dict.pickle', 'wb') as f:
        pickle.dump(time_to_points_dict, f)

    pdb.set_trace()

