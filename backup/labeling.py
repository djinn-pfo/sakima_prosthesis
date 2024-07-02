import pdb
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt

def get_hash(point):
    return str(point[0]) + str(point[1]) + str(point[2])

num_points = 12
point_identity_threshold = 0.05
valid_time_diff = 0.5

for k in range(1, 6):

    # Convert input format to dictionary
    with open(f'./walk_dataset/measurement_20240426/trial_20240426/trial_20240426_0{k}.csv') as fin:
        time_points = {}
        while True:
            line = fin.readline()
            if not line:
                break

            if (not line.split(',')[0].isdigit()):
                continue
            else:
                line = line.rstrip('\n')
                elements = line.split(',')

                sample_id = int(elements[0])
                sample = int(elements[1])
                time = float(elements[2])
                
                points = []
                for i in range(3, len(elements), 3):
                    if elements[i] == '*':
                        break
                    coordinates = np.array([float(elements[i]), float(elements[i+1]), float(elements[i+2])])
                    points.append(coordinates)

                if num_points == len(points):
                    time_points[time] = points

    total_sample_times = sorted(time_points.keys())

    sample_times_list = []
    sample_time = []
    for i in range(len(total_sample_times)-1):
        if (total_sample_times[i+1] - total_sample_times[i]) < valid_time_diff:
            sample_time.append(total_sample_times[i])
        else:
            sample_times_list.append(sample_time)
            sample_time = []

    time_label_points_list = []
    for sample_times in sample_times_list:

        # Extract trajectories
        current_label = 0
        label_trajectories = {}

        initial_time = sample_times[0]
        initial_points = time_points[initial_time]

        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = [point[0] for point in initial_points]
        ys = [point[1] for point in initial_points]
        zs = [point[2] for point in initial_points]

        ax.scatter(xs, ys, zs)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
        """
        
        # Assign initial labels
        for i in range(len(initial_points)):
            point = initial_points[i]
            label_trajectories[current_label] = [{'time': initial_time, 'point': point}]
            current_label += 1

        labels = label_trajectories.keys()
        for i in tqdm.tqdm(range(1, len(sample_times))):

            focused_time = sample_times[i]
            focused_points = time_points[focused_time]

            for label in labels:
                target = label_trajectories[label][-1]['point']
                dists = [np.linalg.norm(focused - target) for focused in focused_points]
                closest_point = focused_points[np.argmin(dists)]

                label_trajectories[label].append({'time': focused_time, 'point': closest_point})

        trajectory_lengths = [len(label_trajectories[label]) for label in label_trajectories.keys()]

        time_label_points = {}
        for label in label_trajectories.keys():
            trajectory = label_trajectories[label]
            for point in trajectory:
                if point['time'] not in time_label_points:
                    time_label_points[point['time']] = [{'point': point['point'], 'label': label}]
                else:
                    time_label_points[point['time']].append({'point': point['point'], 'label': label})
        time_label_points_list.append(time_label_points)

    with open(f'time_label_points_list_{k}.pickle', 'wb') as fout:
        pickle.dump(time_label_points_list, fout)

