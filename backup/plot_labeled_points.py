import pickle
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import os

st.title("(株)佐喜眞義肢 ラベル付き点群データ可視化")

fig = go.Figure()

pickle_directory = './'
pickle_files = [f for f in os.listdir(pickle_directory) if f.endswith('.pickle')]

selected_file = st.selectbox('Select a pickle file:', pickle_files)

# Load labeled points from selected pickle file
if selected_file:
    with open(selected_file, 'rb') as file:
        time_based_dict = pickle.load(file)
else:
    st.error("Please select a file.")

times = sorted(time_based_dict.keys())
points = time_based_dict[times[0]]
# Set the range for each axis
# x_range = [min(point['point'][0] for point in points), max(point['point'][0] for point in points)]
# y_range = [min(point['point'][1] for point in points), max(point['point'][1] for point in points)]
# z_range = [min(point['point'][2] for point in points), max(point['point'][2] for point in points)]
# x_range = [-0.5, -0.9]
z_range = [0, 1.5]

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(range=x_range),
#         yaxis=dict(range=y_range),
#         zaxis=dict(range=z_range)
#     )
# )

fig.update_layout(
    scene=dict(
        zaxis=dict(range=z_range)
    )
)


for point in points:
    fig.add_trace(
        go.Scatter3d(
            x=[point['point'][0]],
            y=[point['point'][1]],
            z=[point['point'][2]],
            mode='markers+text',
            marker=dict(size=2),
            text=[point['label']],  # Add this line to include labels
            textposition='top center'  # Optional: position of the text
        )
    )

# Create frames for the animation
frames = []
for i, time in enumerate(sorted(time_based_dict.keys())):
    frame = go.Frame(
        data=[go.Scatter3d(
            x=[point['point'][0]],
            y=[point['point'][1]],
            z=[point['point'][2]],
            mode='markers+text',
            marker=dict(size=2),
            text=[point['label']],  # Add this line to include labels
            textposition='top center'  # Optional: position of the text
        ) for point in time_based_dict[time]],
        name=str(time),
        layout=go.Layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0)  # Reduce margins to expand plot to full browser width
        )
    )
    frames.append(frame)

fig.frames = frames

play_button = {
    'args': [None, {'frame': {'duration': 1, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 1, 'easing': 'quadratic-in-out'}}],
    'label': 'Play',
    'method': 'animate'
}

pause_button = {
    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
    'label': 'Pause',
    'method': 'animate'
}

fig.update_layout(
    updatemenus=[{
        'buttons': [play_button, pause_button],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Show the figure
st.plotly_chart(fig)
