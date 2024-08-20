import pickle
import pdb
import numpy as np

import matplotlib.pyplot as plt


def main(file_names):

    indices_dict = {
        "calib_180_pre.pickle": [[6, 4], [8, 15], [17, 16, 21, 19, 7], [18]],
        "none_2_20240604.pickle": [[3, 2], [6, 5], [12, 8, 10], [9]],
        "tanka_20240604.pickle": [[3, 2], [4, 5], [12, 6, 16], [10]],
        "cb_tanka_20240604.pickle": [[3, 2], [5, 6], [12, 4, 18], [8]],
        "cb_tanka_2_20240604.pickle": [[3, 2], [15, 6], [11, 4, 16], [7]],
    }

    dist_dict = {}
    upper_angle_dict = {}
    for file_name in file_names:
        with open(file_name, "rb") as f:
            time_points_dict = pickle.load(f)

        timestamps = sorted(list(time_points_dict.keys()))
        indices = indices_dict[file_name]
        label_dist_dict = {}

        upper_ungles = []
        for timestamp in timestamps:
            points = time_points_dict[timestamp]

            left_waist_index = indices[1][1]

            label_point_dict = {}
            for point in points:
                label_point_dict[point["label"]] = point["point"]

            try:
                upper0 = label_point_dict[indices[2][0]]
                upper1 = label_point_dict[indices[2][1]]
                upper2 = label_point_dict[indices[2][2]]
            except:
                break

            upper0_2 = upper0 - upper2
            upper1_2 = upper1 - upper2
            upper0_2 /= np.linalg.norm(upper0_2)
            upper1_2 /= np.linalg.norm(upper1_2)
            upper_ungles.append(np.arccos(np.dot(upper0_2, upper1_2)))

            left_waist = label_point_dict[left_waist_index]
            for index in indices[2]:
                coords = label_point_dict[index]
                dist = np.linalg.norm(coords - left_waist)
                if index not in label_dist_dict:
                    label_dist_dict[index] = [dist]
                else:
                    label_dist_dict[index].append(dist)

        dist_dict[file_name] = label_dist_dict
        upper_angle_dict[file_name] = upper_ungles

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(file_names)))

    for i, file_name in enumerate(file_names):
        label_dist_dict = dist_dict[file_name]
        label_plot = False
        for label, dists in label_dist_dict.items():
            plt.plot(dists[:120], color=colors[i])
        # for label, dists in label_dist_dict.items():
        #     plt.plot(dists[:120], color=colors[i])

    for i, file_name in enumerate(file_names):
        label_dist_dict = dist_dict[file_name]
        for label, dists in label_dist_dict.items():
            plt.text(0, dists[120], label)

    plt.savefig("validation_dist.png")
    plt.close()

    for i, file_name in enumerate(file_names):
        angles = upper_angle_dict[file_name]
        plt.plot(angles[:120], color=colors[i])
    plt.savefig("validation_angle.png")
    plt.close()


if __name__ == "__main__":

    file_names = (
        "calib_180_pre.pickle",
        "none_2_20240604.pickle",
        "tanka_20240604.pickle",
        "cb_tanka_20240604.pickle",
        "cb_tanka_2_20240604.pickle",
    )

    main(file_names)
