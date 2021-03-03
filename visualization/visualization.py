import numpy as np
import plotly.graph_objects as go


def visualize_mask(mask, cube, ndgrid_idxs=slice(0, 1, 50j)):
    filled = np.add(cube, mask)
    x, y, z = np.mgrid[ndgrid_idxs, ndgrid_idxs, ndgrid_idxs]

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        #value=filled.flatten(),
        value=mask.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1,       # needs to be small to see through all surfaces
        surface_count=17,  # needs to be a large number for good volume rendering
    ))
    ax_layout = {
        'visible': False, 
        'showticklabels': False, 
        'showgrid': False, 
        'zeroline': False
    }
    """
    fig.update_layout(scene={
        'xaxis': ax_layout, 'yaxis': ax_layout, 'zaxis': ax_layout
    })
    """
    return fig # fig.show()