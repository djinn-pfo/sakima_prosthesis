import pandas as pd
import pdb
import numpy as np
import sys
import tqdm
import pickle

data = pd.read_csv(sys.argv[1], comment='#')

time_points_dict = {}
for index, row in data.iterrows():
    time = row['Time']
    points = []
    for i in range(1, 14):  # There are 13 points as per the data
        point = (row[f'{i}(X)'], row[f'{i}(Y)'], row[f'{i}(Z)'])
        if '*' not in point:
            point = tuple(float(coord) for coord in point)
            points.append(point)
    time_points_dict[time] = points

filtered_time_points_dict = {time: points for time, points in time_points_dict.items() if len(points) >= 12}
time_points_dict = filtered_time_points_dict

times = sorted(list(time_points_dict.keys()))

# Identify consecutive time points and store them in times_list
times_list = []
if times:
    previous_time = times[0]
    consecutive_times = [previous_time]

    for current_time in times[1:]:
        if current_time - previous_time <= 0.00834:  # Assuming consecutive if difference is less than or equal to one frame at 120 Hz
            consecutive_times.append(current_time)
        else:
            if len(consecutive_times) > 1:
                times_list.append(consecutive_times)
            consecutive_times = [current_time]
        previous_time = current_time

    # Add the last set of consecutive times if it's more than one
    if len(consecutive_times) > 1:
        times_list.append(consecutive_times)

# Filter out times_list elements with length less than or equal to 1000
times_list = [consecutive_times for consecutive_times in times_list if len(consecutive_times) > 1000]

# Iterate through each list of consecutive times
for consecutive_times in times_list:
    # Initialize a dictionary to store labels for each point at each time within the consecutive times
    labels_dict = {}

    # Initialize label counter
    label_counter = 1

    # Iterate through each time point in the consecutive times list
    for i in tqdm.tqdm(range(len(consecutive_times) - 1)):
        current_time = consecutive_times[i]
        next_time = consecutive_times[i + 1]
        
        # Retrieve points for current and next time
        current_points = time_points_dict[current_time]
        next_points = time_points_dict[next_time]
        
        # Initialize labels for the current time if not already initialized
        if current_time not in labels_dict:
            labels_dict[current_time] = [0] * len(current_points)
        
        # Initialize labels for the next time
        labels_dict[next_time] = [0] * len(next_points)
        
        # Assign labels to current points if not already labeled
        for j, point in enumerate(current_points):
            if labels_dict[current_time][j] == 0:
                labels_dict[current_time][j] = label_counter
                label_counter += 1
        
        # Find the closest current point for each next point and assign the label
        for j, next_point in enumerate(next_points):
            min_distance = float('inf')
            closest_label = None
            for k, current_point in enumerate(current_points):
                distance = np.linalg.norm(np.array(next_point) - np.array(current_point))
                if distance < min_distance:
                    min_distance = distance
                    closest_label = labels_dict[current_time][k]
            labels_dict[next_time][j] = closest_label

    # Convert labels_dict to a dictionary with labels as keys and a list of dictionaries with time and coordinates as values
    output_dict = {}
    for time, labels in labels_dict.items():
        points = time_points_dict[time]
        for label, point in zip(labels, points):
            if label not in output_dict:
                output_dict[label] = []
            output_dict[label].append({'time': time, 'coords': point})
    # Save the output_dict to a file named tmp.pickle
    with open('tmp.pickle', 'wb') as file:
        pickle.dump(output_dict, file)

    # Debugging point to check labels assignment
    pdb.set_trace()

