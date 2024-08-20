import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main(file_name):

    with open(file_name, "rb") as f:
        time_points_dict = pickle.load(f)

    timestamps = sorted(time_points_dict.keys())

    print(file_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    for timestamp in timestamps:

        for point in time_points_dict[timestamp]:
            ax.text(
                point["point"][0],
                point["point"][1],
                point["point"][2],
                str(point["label"]),
                color="black",
            )
            ax.scatter(
                point["point"][0],
                point["point"][1],
                point["point"][2],
                color="blue",
            )
        plt.show()
        break


if __name__ == "__main__":

    file_names = (
        "calib_180_pre.pickle",
        "none_2_20240604.pickle",
        "tanka_20240604.pickle",
        "cb_tanka_20240604.pickle",
        "cb_tanka_2_20240604.pickle",
    )

    for file_name in file_names:
        main(file_name)
