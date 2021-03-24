import numpy as np
import plotly.graph_objects as go


def visualize_mask(filled, ndgrid_idxs=slice(0, 1, 50j)):
    vol = create_volume(filled, ndgrid_idxs)
    
    fig = go.Figure(data=vol)
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
    return fig


def create_volume(data, ndgrid_idxs, colorscale='Viridis', opacity=0.1, n_surface=17):
    x, y, z = np.mgrid[ndgrid_idxs, ndgrid_idxs, ndgrid_idxs]
    return go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=data.flatten(),
        colorscale=colorscale,
        isomin=0.1,
        isomax=0.8,
        opacity=opacity,
        surface_count=n_surface,  # needs to be a large number for good volume rendering
    )