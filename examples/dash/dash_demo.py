
# %%
from pathlib import Path

from dash import Dash, html, dcc, callback, Output, Input, ctx
import dash_bootstrap_components as dbc
import numpy as np
import jax.random as jr
import plotly.io as pio

from feedbax import load, tree_take
import feedbax.plotly as fbp

from dash_demo_data_gen import setup, hyperparameters


data_path = Path(__file__).parent.resolve() / "dash_demo_data.eqx"

task, models = load(data_path, setup)

# %%
key_eval = jr.PRNGKey(0)
n_replicates = hyperparameters['n_replicates']
states = task.eval_ensemble(models, n_replicates, key_eval)

states_0 = tree_take(states, 0)

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Plotly theme
pio.templates.default = "plotly_white"

# Layout settings for all plots
layout_kws = dict(
    margin=dict(l=10, r=10, t=30, b=10),
)
# Horizontal rule style
hr_style={"border-top": '2px solid #c1c1c1'}

app.layout = dbc.Container([
    html.Div(
        children=[
            html.Label('Model replicate'),
            dcc.Slider(
                1,
                n_replicates,
                step=1,
                id="replicate--slider",
                value=0,
            ),
        ],
    ),
    html.Hr(style=hr_style),
    dbc.Row([
        dbc.Col([
            html.H5('Kinematics'),
            trajectory_graph := dcc.Graph(
                id='trajectory-graph',
                figure=fbp.effector_trajectories(states_0),
            ),
        ], width=10),
        dbc.Col([
            html.Div(children=[
                html.P(
                    "Click on data to highlight individual trials across plots.",
                    style=dict(color='gray'),
                ),
                html.P(
                    "Click or double-click on legend entries to control "
                    "absolute visibility of trial data on each plot.",
                    style=dict(color='gray'),
                ),
                dbc.Button(
                    ['Clear', html.Br(), 'highlight'],
                    id="unselect--button",
                    n_clicks=0,
                    color="secondary",
                    outline=True,
                    style=dict(marginTop='1em'),
                ),    
            ]),
            # html.Button('Unselect', id='unselect--button', n_clicks=0),
        ], width=2, className="info_col box"),
    ], align="top"),
    html.Hr(style=hr_style),
    dbc.Row([
        dbc.Col([
            html.H5('Activities of example network units'),
            unit_samples_graph := dcc.Graph(
                id='unit-samples-graph',
                figure=fbp.activity_sample_units(
                    states_0.net.hidden,
                    n_samples=5,
                    key=key_eval,
                ),
            ),
        ], width=6),
        dbc.Col([
            html.H5('Single-trial activity of all network units'),
            units_heatmap_graph := dcc.Graph(
                id='units-heatmap-graph',
                figure=fbp.activity_heatmap(
                    states_0.net.hidden[0],
                ),
            ),
            html.Div(
                children=[
                    html.Label('Trial'),
                    dcc.Slider(
                        1,
                        task.n_validation_trials,
                        step=1,
                        id="trial--slider",
                        value=0,
                    ),
                ],
                # style=dict(display='flex'),
            ),
        ], width=6),
    ]),
    dcc.Store(id='selected-trial-idx', data=None),
], fluid=True)

@callback(
    Output(trajectory_graph, 'figure'),
    Output(unit_samples_graph, 'figure'),
    Output(units_heatmap_graph, 'figure'),
    Output('trial--slider', 'value'),
    Output('selected-trial-idx', 'data'),
    Input(trajectory_graph, 'figure'),
    Input(unit_samples_graph, 'figure'),
    Input(units_heatmap_graph, 'figure'),
    Input(trajectory_graph, 'clickData'),
    Input(unit_samples_graph, 'clickData'),
    Input('replicate--slider', 'value'),
    Input('trial--slider', 'value'),
    Input('unselect--button', 'n_clicks'),
    Input('selected-trial-idx', 'data'),
)
def update(
    fig_traj,
    fig_unit_samples,
    fig_units_heatmap,
    fig_traj_click_data,
    fig_unit_samples_click_data,
    replicate,
    i_trial,
    unselect_n_clicks,
    selected_trial_idx,
):
    states_i = tree_take(states, replicate - 1)

    fig_traj = fbp.effector_trajectories(
        states_i,
        layout_kws=layout_kws | dict(height=350),
    )
    fig_unit_samples = fbp.activity_sample_units(
        states_i.net.hidden,
        n_samples=5,
        key=key_eval,
        layout_kws=layout_kws,
    )
    
    trial_name = str(i_trial)

    # Highlight individual trials if clicked on in figures, or with slider.
    # Remove highlight if "Unselect trial" button is clicked.
    if ctx.triggered_id == 'unselect--button':
        fig_traj.update_traces(opacity=1)
        fig_unit_samples.update_traces(opacity=1)
        selected_trial_idx = None
        
    elif ctx.triggered_id == 'trial--slider':
        selected_trial_idx = int(trial_name)
        
    elif ctx.triggered_id == 'replicate--slider':
        if selected_trial_idx is not None:
            trial_name = str(selected_trial_idx)    
               
    elif "clickData" in ctx.triggered[0]['prop_id']:
        curve_number = ctx.triggered[0]['value']['points'][0]['curveNumber']
        fig_clicked = {
            'trajectory-graph': fig_traj,
            'unit-samples-graph': fig_unit_samples,
        }[str(ctx.triggered_id)]
        trial_name = fig_clicked.data[curve_number].name
        selected_trial_idx = int(trial_name)
        
    if selected_trial_idx is not None:
        fig_traj.update_traces(
            opacity=(unselected_opacity := 0.2),
            selector=lambda trace: trace.name != trial_name,
        )
        fig_unit_samples.update_traces(
            opacity=unselected_opacity,
            selector=lambda trace: trace.name != trial_name,
        )
        fig_unit_samples.update_traces(
            line_width=3,
            opacity=1,
            selector=lambda trace: trace.name == trial_name,
        )

    fig_units_heatmap = fbp.activity_heatmap(
        states_0.net.hidden[i_trial],
        layout_kws=layout_kws,
    )
    
    i_trial = selected_trial_idx if selected_trial_idx is not None else i_trial

    return (
        fig_traj, 
        fig_unit_samples, 
        fig_units_heatmap, 
        i_trial, 
        selected_trial_idx,
    )


if __name__ == '__main__':
    app.run(debug=True)


# data = np.stack([
#     states_i.mechanics.effector.pos,
#     states_i.mechanics.effector.vel,
#     states_i.efferent.output,
# ], axis=1)

# data = data.reshape(-1, *data.shape[2:])

# for i, curve in enumerate(data):
#     fig['data'][i]['x'] = curve[..., 0]
#     fig['data'][i]['y'] = curve[..., 1]
