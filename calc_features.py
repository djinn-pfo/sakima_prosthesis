import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pdb


def main(file_names):
    indices_dict = {
        "calib_180_pre.pickle": [[6, 4], [8, 15], [17, 16, 19], [3, 18]],
        "none_2_20240604.pickle": [[3, 2], [6, 5], [12, 8, 10], [17, 9]],
        "tanka_20240604.pickle": [[3, 2], [4, 5], [12, 6, 16], [20, 10]],
        "cb_tanka_20240604.pickle": [[3, 2], [5, 6], [12, 4, 18], [20, 8]],
        "cb_tanka_2_20240604.pickle": [[3, 2], [15, 6], [11, 4, 16], [18, 7]],
    }

    """
    indices_dict = {
        "calib_180_pre.pickle": [[6, 4], [8, 15], [17, 16], [3, 18]],
        "none_2_20240604.pickle": [[3, 2], [6, 5], [12, 8], [17, 9]],
        "tanka_20240604.pickle": [[3, 2], [4, 5], [12, 6], [20, 10]],
        "cb_tanka_20240604.pickle": [[3, 2], [5, 6], [12, 4], [20, 8]],
        "cb_tanka_2_20240604.pickle": [[3, 2], [15, 6], [11, 4], [18, 17]],
    }
    """

    body_axis_angles_dict = {}
    shoulder_centers_dict = {}
    joint_angles_dict = {}
    joint_angles_projected_dict = {}
    for file_name in file_names:

        with open(file_name, "rb") as f:
            time_points_dict = pickle.load(f)

        indices = indices_dict[file_name]

        timestamps = sorted(list(time_points_dict.keys()))

        body_axis_angles = []
        shoulder_centers = []
        joint_angles = []
        joint_angles_projected = []
        for timestamp in timestamps:
            points = time_points_dict[timestamp]

            if len(points) > 17:

                label_point_dict = {}
                for point in points:
                    label_point_dict[point["label"]] = point["point"]

                shoulder_indices = indices[0]
                shoulder_points = [
                    label_point_dict[index] for index in shoulder_indices
                ]
                shoulder_center = np.mean(shoulder_points, axis=0)
                shoulder_centers.append(shoulder_center)

                waist_indices = indices[1]
                try:
                    waist_points = [label_point_dict[index] for index in waist_indices]
                except:
                    pdb.set_trace()
                left_waist = label_point_dict[waist_indices[1]]
                waist_center = np.mean(waist_points, axis=0)

                z_axis = np.array([0, 0, 1])
                body_axis = shoulder_center - waist_center
                body_axis_angles.append(
                    np.rad2deg(
                        np.arccos(
                            np.dot(z_axis, body_axis)
                            / (np.linalg.norm(z_axis) * np.linalg.norm(body_axis))
                        )
                    )
                )

                upper_indices = indices[2]
                upper = label_point_dict[upper_indices[0]]
                upper_points = [label_point_dict[index] for index in upper_indices]
                upper_center = np.mean(upper_points, axis=0)

                uncle_indices = indices[3]
                uncle = label_point_dict[uncle_indices[0]]
                uncle_points = [label_point_dict[index] for index in uncle_indices]
                uncle_center = np.mean(uncle_points, axis=0)

                # upper_to_waist = left_waist - upper_center
                # upper_to_uncle = uncle_center - upper_center
                upper_to_waist = left_waist - upper
                upper_to_uncle = uncle - upper

                upper_to_waist_projected = upper_to_waist.copy()
                upper_to_uncle_projected = upper_to_uncle.copy()
                upper_to_waist_projected[0] = 0
                upper_to_uncle_projected[0] = 0

                joint_angles.append(
                    np.rad2deg(
                        np.arccos(
                            np.dot(upper_to_waist, upper_to_uncle)
                            / (
                                np.linalg.norm(upper_to_waist)
                                * np.linalg.norm(upper_to_uncle)
                            )
                        )
                    )
                )

                joint_angles_projected.append(
                    np.rad2deg(
                        np.arccos(
                            np.dot(upper_to_waist_projected, upper_to_uncle_projected)
                            / (
                                np.linalg.norm(upper_to_waist_projected)
                                * np.linalg.norm(upper_to_uncle_projected)
                            )
                        )
                    )
                )

        body_axis_angles_dict[file_name] = body_axis_angles
        shoulder_centers_dict[file_name] = shoulder_centers
        joint_angles_dict[file_name] = joint_angles
        joint_angles_projected_dict[file_name] = joint_angles_projected

    for file_name in file_names:
        plt.plot(body_axis_angles_dict[file_name])
    plt.show()

    for file_name in file_names:
        shoulder_centers = shoulder_centers_dict[file_name]
        times = range(len(shoulder_centers))
        shoulder_centers = np.array(shoulder_centers)

        plt.scatter(
            shoulder_centers[:, 0], shoulder_centers[:, 2], c=times, cmap="viridis"
        )
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.title(f"Scatter plot of shoulder centers for {file_name}")
    plt.show()

    for file_name in file_names:
        plt.ylim(140, 180)
        plt.plot(joint_angles_dict[file_name])
    plt.show()

    for file_name in file_names:
        plt.plot(joint_angles_projected_dict[file_name])
    plt.show()


if __name__ == "__main__":

    file_names = (
        "calib_180_pre.pickle",
        "none_2_20240604.pickle",
        "tanka_20240604.pickle",
        "cb_tanka_20240604.pickle",
        "cb_tanka_2_20240604.pickle",
    )

    main(file_names)
