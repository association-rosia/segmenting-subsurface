import numpy as np
import plotly.graph_objects as go


def get_volume_hollow(volume: np.ndarray):
    # Keep external layer
    coordinates_tuples = []
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                if x == 0 or x == volume.shape[0] - 1 or y == 0 or y == volume.shape[1] - 1 or z == 0 or z == \
                        volume.shape[2] - 1:
                    x_coord = x
                    y_coord = y
                    z_coord = z
                    coordinates_tuples.append([x_coord, y_coord, z_coord, volume[x, y, z]])

    volume_hollow = np.asarray(coordinates_tuples)

    return volume_hollow


def get_plotly_volume(volume_hollow, colorscale):
    data = go.Scatter3d(
        x=volume_hollow[:, 0], y=volume_hollow[:, 1], z=volume_hollow[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=volume_hollow[:, 3],
            colorscale=colorscale,
            opacity=1),
        hoverinfo='skip'
    )

    return data
