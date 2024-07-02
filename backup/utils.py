from indices import point_indices
from indices import time_indices
import numpy as np
import pickle

def convert_to_time_x_dict(data_dict):
    ks = data_dict.keys()
    directions = ('forward', 'backward')

    time_coords_mats_dict = {}
    time_point_labels_dict = {}

    for k in ks:
        time_coords_mats_dict[k] = {}
        time_point_labels_dict[k] = {}
        for direction in directions:
            time_label_points_list = data_dict[k]
            sample_times = sorted(list(time_label_points_list[time_indices[k][direction]].keys()))

            time_coords_mats = {}
            time_point_labels = {}
            for sample_time in sample_times:
                points = time_label_points_list[time_indices[k][direction]][sample_time]
                coord_mat = []
                point_labels = []
                for point in points:
                    coord_mat.append(point['point'])
                    point_labels.append(point['label'])
                coord_mat = np.array(coord_mat)
                time_coords_mats[sample_time] = coord_mat
                time_point_labels[sample_time] = point_labels
            time_coords_mats_dict[k][direction] = time_coords_mats
            time_point_labels_dict[k][direction] = time_point_labels

    return {'time_coords_mats': time_coords_mats_dict, 'time_point_labels': time_point_labels_dict}

def load_dataset(ks):
    data_dict = {}
    for k in ks:
        with open(f'time_label_points_list_{k}.pickle', 'rb') as fin:
            time_label_points_list = pickle.load(fin)
        data_dict[k] = time_label_points_list

    return data_dict

def get_target_points(time_label_points, k, direction):

    parts = ('shoulder', 'upper', 'joint', 'lower')
    sides = ('left', 'right')

    target_points = {}
    for part in parts:
        target_points[part] = {}
        for side in sides:
            target_points[part][side] = []
            for timestamp, points in time_label_points.items():
                for point in points:
                    if point['label'] == point_indices[k][part][side][direction]:
                        target_points[part][side].append(point['point'])

    return target_points

def get_joint_angles(target_points):

    sides = ('left', 'right')

    side_thetas = {}
    side_anta = {}
    for side in sides:
        upper_vecs = target_points['upper'][side]
        joint_vecs = target_points['joint'][side]
        lower_vecs = target_points['lower'][side]
        side_thetas[side] = []
        side_anta[side] = []
        for i in range(len(joint_vecs)):
            joint_upper_vec = upper_vecs[i] - joint_vecs[i]
            joint_lower_vec = lower_vecs[i] - joint_vecs[i]

            theta = np.arccos(np.dot(joint_upper_vec, joint_lower_vec) / (np.linalg.norm(joint_upper_vec) * np.linalg.norm(joint_lower_vec)))
            cross = np.cross(joint_upper_vec, joint_lower_vec)
            if cross[0] < 0:
                side_anta[side].append(1)
            else:
                side_anta[side].append(0)

            side_thetas[side].append(theta)

    return {'angles': side_thetas, 'antas': side_anta}

def get_centers(target_points, part):

    centers = []
    left_vecs = target_points[part]['left']
    right_vecs = target_points[part]['right']
    for i in range(len(left_vecs)):
        center_vec = (left_vecs[i] + right_vecs[i]) / 2
        centers.append(center_vec)

    return centers
