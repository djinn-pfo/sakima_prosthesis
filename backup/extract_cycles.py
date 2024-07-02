from utils import get_target_points, get_joint_angles, get_centers, load_dataset
from indices import time_indices
import pdb
import numpy as np
import pickle

# Load data
data_dict = load_dataset([1, 2, 3, 4, 5])

cycle_tips_dict = {}
for k in data_dict.keys():
    cycle_tips_dict[k] = {}
    time_label_points_list = data_dict[k]

    directions = ('forward', 'backward')
    sides = ('left', 'right')
    for direction in directions:
        time_label_points = time_label_points_list[time_indices[k][direction]]

        target_points = get_target_points(time_label_points, k, direction)
        angles_dict = get_joint_angles(target_points)
        angles_right = np.array(angles_dict['angles']['right'])
        angles_left = np.array(angles_dict['angles']['left'])
        diffs = np.diff(angles_right)
        prod_diffs = []
        for i in range(len(diffs) - 1):
            if diffs[i]*diffs[i + 1] < 0:
                prod_diffs.append(-1)
            else:
                prod_diffs.append(1)
        prod_diffs = np.array(prod_diffs)
        subangles_right = angles_right[1:-1]
        
        max_bend_timing_candidates = np.where((prod_diffs == -1) & (subangles_right < 2.3))[0]

        max_bend_timings = [max_bend_timing_candidates[0]]
        for i in range(1, len(max_bend_timing_candidates)):
            if 100 < max_bend_timing_candidates[i] - max_bend_timings[-1]:
                max_bend_timings.append(max_bend_timing_candidates[i])

        subangles_left = angles_left[1:-1]
        cycle_tips = []
        for i in range(len(max_bend_timings) - 1):
            start = max_bend_timings[i]
            end = max_bend_timings[i + 1]
            focused_angles = angles_left[start:end]
            tip = np.argmax(focused_angles)
            cycle_tips.append(start + tip)
        
        cycle_tips_dict[k][direction] = cycle_tips

pdb.set_trace()

with open('cycle_tips_dict.pickle', 'wb') as fout:
    pickle.dump(cycle_tips_dict, fout)


