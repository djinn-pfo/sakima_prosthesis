import pickle
import numpy as np
import pdb
import matplotlib.pyplot as plt

from indices import point_indices, time_indices
from utils import get_target_points, get_joint_angles, get_centers, load_dataset

# Load data
data_dict = load_dataset([1, 2, 3, 4, 5])

left_thetas = []
right_thetas = []
shoulder_centers = []
body_centers = []
body_center_angles = []

for k in data_dict.keys():
    time_label_points_list = data_dict[k]

    directions = ('forward', 'backward')
    sides = ('left', 'right')
    for direction in directions:
        time_label_points = time_label_points_list[time_indices[k][direction]]

        target_points = get_target_points(time_label_points, k, direction)
        angles_dict = get_joint_angles(target_points)
        angles = angles_dict['angles']
        antas = angles_dict['antas']

        shoulder_centers = get_centers(target_points, 'shoulder')
        upper_centers = get_centers(target_points, 'upper')

        body_center_angles = []
        for i in range(len(shoulder_centers)):
            body_center = shoulder_centers[i] - upper_centers[i]

            body_center_arrow = body_center
            body_center_arrow[0] = 0
            body_center_ground = body_center_arrow - np.array([0, 0, 100])
            body_center_ground[2] = 0

            body_center_angles.append(np.arccos(np.dot(body_center_arrow, body_center_ground) / (np.linalg.norm(body_center_arrow) * np.linalg.norm(body_center_ground))))

        plt.plot(angles['left'])
        plt.plot(antas['left'])
        plt.plot(angles['right'])
        plt.savefig(f'joint_angle_{k}_{direction}.png')
        plt.close()

        plt.plot(body_center_angles)
        plt.savefig(f'body_center_angle_{k}_{direction}.png')
        plt.close()

        shoulder_centers = np.array(shoulder_centers)
        plt.scatter(shoulder_centers[:, 0], shoulder_centers[:, 2], s=1)
        plt.savefig(f'shoulder_center_{k}_{direction}.png')
        plt.close()

