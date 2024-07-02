import os
import glob
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

from utils import extract_trajectory_reverse, linear_interpolate_trajectory, load_points_from_csv, extract_available_regions, check_point_identity

def main():

    valid_frames_dict = {
        'test_measurement_20240604.csv': 420
    }

    # Specify the directory containing the CSV files
    directory_path = 'walk_dataset/measurement_20240604/'

    # List all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    # Loop through each CSV file and process it
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        filename = csv_file.split('/')[-1]

        time_points_dict = load_points_from_csv(csv_file)

        times = np.array(sorted(list(time_points_dict.keys())))
        point_nums = np.array([len(time_points_dict[time]) for time in times])

        available_regions = extract_available_regions(point_nums, 0.9)
        region_lengths = np.array([len(region) for region in available_regions])
        focused_indices = available_regions[np.argmax(region_lengths)]
        focused_times = times[focused_indices]

        last_available_index = np.max(np.where(point_nums[focused_indices] == np.min(point_nums[focused_indices]))[0])
        last_available_time = focused_times[last_available_index]

        initial_time = last_available_time
        initial_points = time_points_dict[initial_time]
        trajectories = {}
        label = 1
        for point in initial_points:

            trajectory = extract_trajectory_reverse(point, focused_times, time_points_dict)
            # interpolated_trajectory = linear_interpolate_trajectory(trajectory)
            trajectories[label] = trajectory

            label += 1

        """
        fig = plt.figure()
        ax_trajectory = fig.add_subplot(121, projection='3d')
        ax_displacements = fig.add_subplot(122)

        color_map = plt.cm.rainbow(np.linspace(0, 1, len(trajectories.keys())))
        for i, label in enumerate(trajectories.keys()):
            points = np.array([point['point'] for point in trajectories[label]])
            ax_trajectory.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color_map[i]]*len(points), marker='+')

            displacements = []
            for i in range(len(points)-1):
                displacement = np.linalg.norm(points[i, 0:2] - points[i+1, 0:2])
                displacements.append(displacement)

            ax_displacements.plot(displacements)

            # png_file = filename.replace('.csv', f'_{label}.png')
            # plt.savefig(png_file)
            # plt.close()
        plt.show()
        """

        labeled_time_points_dict = {}
        for label, points in trajectories.items():
            for point_info in points:
                time = point_info['time']
                point = point_info['point']
                if time not in labeled_time_points_dict:
                    labeled_time_points_dict[time] = []
                labeled_time_points_dict[time].append({'label': label, 'point': point})

        times = sorted(list(labeled_time_points_dict.keys()))
        for time in times:
            for i in range(len(labeled_time_points_dict[time]) - 1):
                if check_point_identity(labeled_time_points_dict[time][i]['point'], labeled_time_points_dict[time][i + 1]['point']):
                    print(f"Duplicate point at time {time}/{times[-1]}")

        filename = filename.replace('.csv', '.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(labeled_time_points_dict, f)

if __name__ == '__main__':
    main()
