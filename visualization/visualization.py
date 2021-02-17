import numpy as np
import plotly.graph_objects as go


cube = np.ones(shape=(50, 50, 50))
cube = np.multiply(cube, 0.1)
mask = np.copy(cube)
mask = np.multiply(mask, 0)
mask[18:20, 19:21, 12:15] = 1
filled = np.add(cube, mask)

x, y, z = np.mgrid[0:1:50j, 0:1:50j, 0:1:50j]

fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=filled.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1,       # needs to be small to see through all surfaces
    surface_count=17,  # needs to be a large number for good volume rendering
    ))

fig.update_layout(
    scene={'xaxis': {'visible': False, 'showticklabels': False, 'showgrid': False, 'zeroline': False},
           'yaxis': {'visible': False, 'showticklabels': False, 'showgrid': False, 'zeroline': False},
           'zaxis': {'visible': False, 'showticklabels': False, 'showgrid': False, 'zeroline': False}})

fig.show()
