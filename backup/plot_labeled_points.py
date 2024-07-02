import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np

from indices import point_indices, time_indices
from utils import get_target_points, load_dataset, convert_to_time_x_dict, get_centers

import pdb

st.title('(株)佐喜眞義肢 動作簡易プロット')

data_dict = load_dataset([1, 2, 3, 4, 5])

time_x_dicts = convert_to_time_x_dict(data_dict)
time_coords_mats_dict = time_x_dicts['time_coords_mats']
time_point_labels_dict = time_x_dicts['time_point_labels']

pattern = st.selectbox('Select pattern', ('1', '2', '3', '4', '5'))
direction = st.selectbox('Select direction', ('forward', 'backward'))
time_index = st.slider("Select time", 0, len(time_coords_mats_dict[int(pattern)][direction]), 1)

time_coords_mats = time_coords_mats_dict[int(pattern)][direction]
time_point_labels = time_point_labels_dict[int(pattern)][direction]
sample_times = sorted(list(time_coords_mats.keys()))
time_points = data_dict[int(pattern)][time_indices[int(pattern)][direction]]
target_points = get_target_points(time_points, int(pattern), direction)

shoulder_centers = get_centers(target_points, 'shoulder')
upper_centers = get_centers(target_points, 'upper')

x = np.linspace(-3, 3, 100)
y = np.linspace(0, 10, 100)
z = np.zeros()
x, y = np.meshgrid(x, y)
pdb.set_trace()
# Create a 3D scatter plot animation using Plotly
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=time_coords_mats[sample_times[time_index]][:, 0],
            y=time_coords_mats[sample_times[time_index]][:, 1],
            z=time_coords_mats[sample_times[time_index]][:, 2],
            mode='markers',
            marker=dict(size=2, color='darkgreen'),
            text=time_point_labels[sample_times[time_index]],
            textposition='top center'
        ),
        go.Scatter3d(
            x=[shoulder_centers[time_index][0], upper_centers[time_index][0]],
            y=[shoulder_centers[time_index][1], upper_centers[time_index][1]],
            z=[shoulder_centers[time_index][2], upper_centers[time_index][2]],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        go.Scatter3d(
            x=[time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['upper']['left'][direction]][0], 
               time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['joint']['left'][direction]][0]],
            y=[time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['upper']['left'][direction]][1], 
               time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['joint']['left'][direction]][1]],
            z=[time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['upper']['left'][direction]][2], 
               time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['joint']['left'][direction]][2]],
            mode='lines',
            line=dict(color='blue', width=2)
        ), 
        go.Scatter3d(
            x=[time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['lower']['left'][direction]][0], 
               time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['joint']['left'][direction]][0]],
            y=[time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['lower']['left'][direction]][1], 
               time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['joint']['left'][direction]][1]],
            z=[time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['lower']['left'][direction]][2], 
               time_coords_mats[sample_times[time_index]][point_indices[int(pattern)]['joint']['left'][direction]][2]],
            mode='lines',
            line=dict(color='blue', width=2)
        )
    ],
    layout=go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0, y=1),
        scene=dict(
            bgcolor='gray',
        )
    )
)

st.plotly_chart(fig)
