import pickle
import pdb
import numpy as np

import matplotlib.pyplot as plt


def main(file_names):

    indices_dict = {
        "calib_180_pre.pickle": [[6, 4], [8, 15], [17, 16, 19], [3, 18]],
        "none_2_20240604.pickle": [[3, 2], [6, 5], [12, 8, 10], [17, 9]],
        "tanka_20240604.pickle": [[3, 2], [4, 5], [12, 6, 16], [20, 10]],
        "cb_tanka_20240604.pickle": [[3, 2], [5, 6], [12, 4, 18], [20, 8]],
        "cb_tanka_2_20240604.pickle": [[3, 2], [15, 6], [11, 4, 16], [18, 7]],
    }

    dist_dict = {}
    angle_dict = {}
    for file_name in file_names:
        with open(file_name, "rb") as f:
            time_points_dict = pickle.load(f)

        timestamps = sorted(list(time_points_dict.keys()))
        indices = indices_dict[file_name]
        label_dist_dict = {}

        upper_ungles = []
        for timestamp in timestamps:
            points = time_points_dict[timestamp]
            if len(points) > 17:
                left_waist_index = indices[1][1]

                label_point_dict = {}
                for point in points:
                    label_point_dict[point["label"]] = point["point"]

                upper0 = label_point_dict[indices[2][0]]
                upper1 = label_point_dict[indices[2][1]]
                try:
                    upper2 = label_point_dict[indices[2][2]]
                except:
                    pdb.set_trace()

                upper0_1 = upper0 - upper1
                upper2_1 = upper2 - upper1
                upper_ungles.append(
                    np.arccos(np.dot(upper0_1, upper2_1))
                    / (np.linalg.norm(upper0_1) * np.linalg.norm(upper2_1))
                )

                left_waist = label_point_dict[left_waist_index]
                for label, coords in label_point_dict.items():
                    dist = np.linalg.norm(left_waist - coords)
                    if label not in label_dist_dict:
                        label_dist_dict[label] = [dist]
                    else:
                        label_dist_dict[label].append(dist)

        dist_dict[file_name] = label_dist_dict
        angle_dict[file_name] = upper_ungles

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(file_names)))

    for i, file_name in enumerate(file_names):
        label_dist_dict = dist_dict[file_name]
        for label, dists in label_dist_dict.items():
            plt.plot(dists[:120], color=colors[i])
    plt.savefig("validation_dist.png")
    plt.close()

    for i, file_name in enumerate(file_names):
        angles = angle_dict[file_name]
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
