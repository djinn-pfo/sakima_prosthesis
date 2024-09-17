import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pdb
from sklearn.decomposition import PCA
from matplotlib import font_manager


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

    label_dict = {
        "calib_180_pre.pickle": "最大伸展位置確認",
        "none_2_20240604.pickle": "装具なし",
        "tanka_20240604.pickle": "AFO",
        "cb_tanka_20240604.pickle": "CB + AFO, 1",
        "cb_tanka_2_20240604.pickle": "CB + AFO, 2",
    }

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
            neg_upper_to_waist = -1 * upper_to_waist
            upper_to_uncle = uncle - upper

            neg_upper_to_waist_projected = neg_upper_to_waist.copy()
            upper_to_uncle_projected = upper_to_uncle.copy()
            neg_upper_to_waist_projected[0] = 0
            upper_to_uncle_projected[0] = 0

            joint_angles.append(
                np.rad2deg(
                    np.arccos(
                        np.dot(neg_upper_to_waist, upper_to_uncle)
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
                        np.dot(neg_upper_to_waist_projected, upper_to_uncle_projected)
                        / (
                            np.linalg.norm(neg_upper_to_waist_projected)
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

        plt.scatter(state_vector_pca[:, 0], state_vector_pca[:, 1])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"PCA of state vectors for {file_name}")
        outfile = file_name.replace(".pickle", "_pca_plot.png")
        plt.savefig(outfile)

        skip_length = 10
        state_vector_2d = state_vector_pca[:, :2]
        diff_state_vector_2d = np.array(
            [
                state_vector_2d[i + skip_length] - state_vector_2d[i]
                for i in range(len(state_vector_2d) - skip_length)
            ]
        )
        norms = np.array([np.linalg.norm(vec) for vec in diff_state_vector_2d])
        state_thetas = np.array(
            [
                diff_state_vector_2d[i, 0] / norms[i]
                for i in range(len(diff_state_vector_2d))
            ]
        )

        ticks = []
        threshold = 0.0
        for i in range(len(state_thetas) - 1):
            if state_thetas[i] > threshold and state_thetas[i + 1] < threshold:
                ticks.append(i)
        ticks_dict[file_name] = ticks

        if file_name != "calib_180_pre.pickle":
            plt.plot(state_thetas)
            plt.plot([threshold] * len(state_thetas))
            outfile = file_name.replace(".pickle", f"_thetas.png")
            plt.savefig(outfile)
            plt.close()

    font_path = "C:\\Users\\hitos\\Downloads\\Noto_Sans_JP\\static\\NotoSansJP-Regular.ttf"

    plt.rcParams["font.family"] = "Noto Sans JP"
    font_manager.fontManager.addfont(font_path)


    # Plot the angle between body-axis and z-axis among whole walking-phase.
    for file_name in file_names:
        # label = file_name.replace(".pickle", "")
        if file_name != "cb_tanka_2_20240604.pickle":
            label = label_dict[file_name]
            plt.plot(body_axis_angles_dict[file_name], label=label)
    plt.legend()
    plt.title("全ケースにおける垂直軸に対する体幹軸角度")
    plt.savefig("body_axis_angles_whole.png")
    plt.close()

    # Plot the angle between body-axis and z-axis for each walking-phase
    for file_name in file_names:
        if file_name != "calib_180_pre.pickle":
            label = label_dict[file_name]
            ticks = ticks_dict[file_name]
            # for i in range(len(ticks) - 1):
            body_axis_maxs = []
            body_axis_mins = []
            for i in range(1, 3):
                plt.plot(
                    body_axis_angles_dict[file_name][ticks[i] : ticks[i + 1]], label=i
                )
                body_axis_maxs.append(np.max(body_axis_angles_dict[file_name][ticks[i] : ticks[i + 1]]))
                body_axis_mins.append(np.min(body_axis_angles_dict[file_name][ticks[i] : ticks[i + 1]]))
            plt.title(f"ケース{label}における垂直軸に対する体幹軸角度")
            plt.legend()
            outfile = file_name.replace(".pickle", f"_baa_phase.png")
            plt.savefig(outfile)
            plt.close()

            print(f"({file_name}) mean of body_axis_maxs: {np.mean(body_axis_maxs)}")
            print(f"({file_name}) mean of body_axis_mins: {np.mean(body_axis_mins)}")

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
    plt.title(f"全ケースにおける全額面内両肩中心位置")
    plt.savefig("shoulder_center_whole.png")

    # Plot the shoulder center positions for each walking-phase in a png file.
    for file_name in file_names:
        if file_name != "calib_180_pre.pickle":
            label = label_dict[file_name]
            shoulder_centers = shoulder_centers_dict[file_name]
            shoulder_centers = np.array(shoulder_centers)
            ticks = ticks_dict[file_name]
            range_xs = []
            range_zs = []
            areas = []
            # for i in range(len(ticks) - 1):
            for i in range(1, 3):
                xs = shoulder_centers[ticks[i] : ticks[i + 1], 0] \
                    - np.mean(shoulder_centers[ticks[i] : ticks[i + 1], 0]),
                zs = shoulder_centers[ticks[i] : ticks[i + 1], 2] \
                    - np.mean(shoulder_centers[ticks[i] : ticks[i + 1], 2]),
                plt.scatter(
                    xs,
                    zs,
                    c=range(ticks[i + 1] - ticks[i]),
                    cmap="viridis",
                    s=2,
                )
                plt.xlabel("X")
                plt.ylabel("Z")
                plt.title(f"ケース{label}における全額面内両肩中心位置")

                range_xs.append(np.max(xs) - np.min(xs))
                range_zs.append(np.max(zs) - np.min(zs))
                areas.append(range_xs[-1] * range_zs[-1] * 100 * 100)

            print(f"({file_name}) mean of areas: {np.mean(areas)}")

            outfile = file_name.replace(".pickle", f"_sc_phase.png")
            plt.savefig(outfile)
            plt.close()

    # Plot joint angles among whole walking-phase
    for file_name in file_names:
        # label = file_name.replace(".pickle", "")
        label = label_dict[file_name]
        if file_name == "calib_180_pre.pickle":
            zero_level = np.mean(joint_angles_dict[file_name])
            plt.plot([0] * len(joint_angles_dict[file_name]), label=label)
        elif file_name != "cb_tanka_2_20240604.pickle":
            plt.plot(joint_angles_dict[file_name] - zero_level, label=label)
    plt.title("全ケースにおける伸展位からの膝関節角度")
    plt.legend()
    outfile = "joint_angles_whole.png"
    plt.savefig(outfile)
    plt.close()

    cb_tankas = ['calib_180_pre.pickle', 'cb_tanka_20240604.pickle', 'cb_tanka_2_20240604.pickle']
    for file_name in cb_tankas:
        label = label_dict[file_name]
        if file_name == "calib_180_pre.pickle":
            zero_level = np.mean(joint_angles_dict[file_name])
            plt.plot([0] * len(joint_angles_dict[file_name]), label=label)
        else:
            plt.plot(joint_angles_dict[file_name] - zero_level, label=label)
    plt.title("CB+AFOの繰り返しにおける伸展位からの膝関節角度")
    plt.legend()
    outfile = "joint_angles_cb_tankas.png"
    plt.savefig(outfile)
    plt.close()


    # Plot joint angles for each walking-phase
    for file_name in file_names:

        if file_name == "calib_180_pre.pickle":
            zero_level = np.mean(joint_angles_dict[file_name])
            plt.plot([0] * len(joint_angles_dict[file_name]), label=label)
        else:
            label = label_dict[file_name]
            ticks = ticks_dict[file_name]
            # for i in range(len(ticks) - 1):
            joint_angle_maxs = []
            joint_angle_mins = []
            for i in range(1, 3):
                plt.plot(joint_angles_dict[file_name][ticks[i] : ticks[i + 1]] - zero_level, label=i)
                joint_angle_mins.append(
                    np.min(joint_angles_dict[file_name][ticks[i] : ticks[i + 1]] - zero_level)
                )
                joint_angle_maxs.append(
                    np.max(joint_angles_dict[file_name][ticks[i] : ticks[i + 1]] - zero_level)
                )
            plt.title(f"ケース{label}の伸展位からの膝関節角度")
            plt.legend()
            outfile = file_name.replace(".pickle", f"_joint_phase.png")
            plt.savefig(outfile)
            plt.close()

            print(f"({file_name}) mean of joint_angle_maxs: {np.mean(joint_angle_maxs)}")
            print(f"({file_name}) mean of joint_angle_mins: {np.mean(joint_angle_mins)}")

    # Plot joint angles among whole walking-phase in YZ plane
    for file_name in file_names:
        # label = file_name.replace(".pickle", "")
        label = label_dict[file_name]
        if file_name == "calib_180_pre.pickle":
            zero_level = np.mean(joint_angles_dict[file_name])
            plt.plot([0] * len(joint_angles_projected_dict[file_name]), label=label)
        else:
            plt.plot(joint_angles_projected_dict[file_name] - zero_level, label=label)
    plt.title("全ケースにおける矢状面内膝関節角度")
    plt.legend()
    outfile = "joint_angles_whole_projected.png"
    plt.savefig(outfile)
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
