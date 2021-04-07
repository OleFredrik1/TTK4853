import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.graph_objects as go

from dash.dependencies import Input, Output, State


class VisualizationApp(object):
    def __init__(self, data_dict, mesh_xyz):
        nd_slice = slice(0, data_dict['all-activation'].shape[1], 1)
        self.fig_data = {
            name: self._create_volume(data, nd_slice) 
            for name, data in data_dict.items()
        }
        x, y, z = mesh_xyz
        self.fig_data['brain-mesh'] = go.Mesh3d(
            x=x, y=y, z=z, color='lightpink', opacity=0.1, alphahull=4
        )

    @staticmethod
    def _create_volume(data, ndgrid_idxs, opacity=0.1, n_surface=17):
        x, y, z = np.mgrid[ndgrid_idxs, ndgrid_idxs, ndgrid_idxs]
        return go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=data.flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=opacity,
            surface_count=n_surface
        )
        
    def run(self):
        fig = go.Figure()
        fig.add_trace(self.fig_data['inside-activation-squared'])
        
        # Init dash app
        app = dash.Dash(__name__)

        app.layout = html.Div([
            dcc.Graph(
                id="graph", 
                figure=fig, 
                style={'height': '800px'}
            ),
            daq.ToggleSwitch(
                label='Brain contours',
                labelPosition='bottom',
                value=False,
                size=80,
                id='contour-toggle',
            )
        ])
        global fig_data
        fig_data = self.fig_data
        
        @app.callback(
        Output("graph", "figure"), [Input("contour-toggle", "value")]
        )
        def _toggle_contours(toggle_val):
            fig = go.Figure(data=fig_data['inside-activation-squared'])
            if toggle_val:
                fig.add_trace(fig_data['brain-mesh'])
            return fig

        app.run_server(debug=False)