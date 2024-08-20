import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pdb
from sklearn.decomposition import PCA


def main(file_names):
    indices_dict = {
        "calib_180_pre.pickle": [[6, 4], [8, 15], [17, 16, 19], [18]],
        "none_2_20240604.pickle": [[3, 2], [6, 5], [12, 8, 10], [9]],
        "tanka_20240604.pickle": [[3, 2], [4, 5], [12, 6, 16], [10]],
        "cb_tanka_20240604.pickle": [[3, 2], [5, 6], [12, 4, 18], [8]],
        "cb_tanka_2_20240604.pickle": [[3, 2], [15, 6], [11, 4, 16], [7]],
    }

    """
    indices_dict = {
        "calib_180_pre.pickle": [[6, 4], [8, 15], [17, 16, 19], [3, 18]],
        "none_2_20240604.pickle": [[3, 2], [6, 5], [12, 8, 10], [17, 9]],
        "tanka_20240604.pickle": [[3, 2], [4, 5], [12, 6, 16], [20, 10]],
        "cb_tanka_20240604.pickle": [[3, 2], [5, 6], [12, 4, 18], [20, 8]],
        "cb_tanka_2_20240604.pickle": [[3, 2], [15, 6], [11, 4, 16], [18, 7]],
    }
    """

    body_axis_angles_dict = {}
    shoulder_centers_dict = {}
    waist_centers_dict = {}
    upper_centers_dict = {}
    state_vectors_dict = {}
    joint_angles_dict = {}
    joint_angles_projected_dict = {}
    for file_name in file_names:

        with open(file_name, "rb") as f:
            time_points_dict = pickle.load(f)

        indices = indices_dict[file_name]

        timestamps = sorted(list(time_points_dict.keys()))

        body_axis_angles = []
        shoulder_centers = []
        waist_centers = []
        upper_centers = []
        state_vectors = []
        joint_angles = []
        joint_angles_projected = []
        for timestamp in timestamps:
            points = time_points_dict[timestamp]

            label_point_dict = {}
            for point in points:
                label_point_dict[point["label"]] = point["point"]

            shoulder_indices = indices[0]
            shoulder_points = [label_point_dict[index] for index in shoulder_indices]
            shoulder_center = np.mean(shoulder_points, axis=0)
            shoulder_centers.append(shoulder_center)

            try:
                waist_indices = indices[1]
                waist_points = [label_point_dict[index] for index in waist_indices]
            except:
                break

            left_waist = label_point_dict[waist_indices[1]]
            waist_center = np.mean(waist_points, axis=0)
            waist_centers.append(waist_center)

            vertical_axis = np.array([0, 0, 1])
            # vertical_axis = np.array([-waist_center[0], -waist_center[1], 1])
            body_axis = shoulder_center - waist_center
            body_axis_angles.append(
                np.rad2deg(
                    np.arccos(
                        np.dot(vertical_axis, body_axis)
                        / (np.linalg.norm(vertical_axis) * np.linalg.norm(body_axis))
                    )
                )
            )

            upper_indices = indices[2]
            upper = label_point_dict[upper_indices[0]]
            try:
                upper_points = [label_point_dict[index] for index in upper_indices]
                upper_center = np.mean(upper_points, axis=0)
                upper_centers.append(upper_center)
            except:
                break

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

            centered_upper_center = upper_center - waist_center
            centered_shoulder_center = shoulder_center - waist_center

            state_vector = np.array(
                centered_upper_center.tolist() + centered_shoulder_center.tolist()
            )
            state_vectors.append(state_vector)

        body_axis_angles_dict[file_name] = body_axis_angles
        shoulder_centers_dict[file_name] = shoulder_centers
        waist_centers_dict[file_name] = waist_centers
        upper_centers_dict[file_name] = upper_centers
        state_vectors_dict[file_name] = state_vectors
        joint_angles_dict[file_name] = joint_angles
        joint_angles_projected_dict[file_name] = joint_angles_projected

    # Extract the ticks among the walking phases
    ticks_dict = {}
    for file_name in file_names:
        state_vectors = state_vectors_dict[file_name]

        pca = PCA(n_components=2)
        state_vector_pca = pca.fit_transform(state_vectors)

        """
        plt.scatter(state_vector_pca[:, 0], state_vector_pca[:, 1])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"PCA of state vectors for {file_name}")
        plt.show()
        """

        state_vector_2d = state_vector_pca[:, :2]
        diff_state_vector_2d = np.diff(state_vector_2d)

        state_thetas = np.array(
            [
                diff_state_vector_2d[i, 0]
                / (np.linalg.norm(diff_state_vector_2d[i, :]))
                for i in range(len(state_vector_pca))
            ]
        )
        # plt.plot(state_thetas)
        # plt.show()

        ticks = []
        for i in range(len(state_thetas) - 1):
            if state_thetas[i] < 0 and state_thetas[i + 1] > 0:
                ticks.append(i)
        ticks_dict[file_name] = ticks

    # Plot the angle between body-axis and z-axis among whole walking-phase.
    for file_name in file_names:
        label = file_name.replace(".pickle", "")
        plt.plot(body_axis_angles_dict[file_name], label=label)
    plt.legend()
    plt.title("Body Axis Angles among Whole Cases")
    plt.savefig("body_axis_angles_whole.png")
    plt.close()

    # Plot the angle between body-axis and z-axis for each walking-phase
    for file_name in file_names:
        ticks = ticks_dict[file_name]
        for i in range(len(ticks) - 1):
            plt.plot(body_axis_angles_dict[file_name][ticks[i] : ticks[i + 1]], label=i)
        plt.title(f"Body Axis Angle of the Case {file_name}")
        plt.legend()
        outfile = file_name.replace(".pickle", f"_baa_phase.png")
        plt.savefig(outfile)
        plt.close()

    # Plot the shoulder center positions in whole walking-phase
    for file_name in file_names:
        shoulder_centers = shoulder_centers_dict[file_name]
        shoulder_centers = np.array(shoulder_centers)
        plt.scatter(
            shoulder_centers[:, 0],
            shoulder_centers[:, 2],
            c=range(len(shoulder_centers)),
            cmap="viridis",
            s=2,
        )
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"Scatter plot of shoulder centers for {file_name}")
    plt.savefig("shoulder_center_whole.png")

    # Plot the shoulder center positions for each walking-phase
    for file_name in file_names:
        shoulder_centers = shoulder_centers_dict[file_name]
        shoulder_centers = np.array(shoulder_centers)
        ticks = ticks_dict[file_name]
        for i in range(len(ticks) - 1):
            plt.scatter(
                shoulder_centers[ticks[i] : ticks[i + 1], 0]
                - np.mean(shoulder_centers[ticks[i] : ticks[i + 1], 0]),
                shoulder_centers[ticks[i] : ticks[i + 1], 2]
                - np.mean(shoulder_centers[ticks[i] : ticks[i + 1], 2]),
                c=range(ticks[i + 1] - ticks[i]),
                cmap="viridis",
                s=2,
            )
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.title(f"Scatter plot of shoulder centers for {file_name}")
        outfile = file_name.replace(".pickle", f"_sc_phase.png")
        plt.savefig(outfile)
        plt.close()

    # Plot joint angles among whole walking-phase
    for file_name in file_names:
        label = file_name.replace(".pickle", "")
        plt.plot(joint_angles_dict[file_name], label=label)
    plt.legend()
    plt.show()

    """
    for file_name in file_names:
        label = file_name.replace(".pickle", "")
        plt.plot(joint_angles_projected_dict[file_name])
    plt.legend()
    plt.show()
    """


if __name__ == "__main__":

    file_names = (
        "calib_180_pre.pickle",
        "none_2_20240604.pickle",
        "tanka_20240604.pickle",
        "cb_tanka_20240604.pickle",
        "cb_tanka_2_20240604.pickle",
    )

    main(file_names)
