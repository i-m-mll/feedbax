
# %%
from pathlib import Path

from dash import Dash, html, dcc, callback, Output, Input, ctx
import dash_bootstrap_components as dbc
import numpy as np
import jax.random as jr

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
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
    html.Div(className='row', children=[
        html.Div(
            className='one column',
            children=[
                html.Button('Unselect trial', id='unselect--button', n_clicks=0),
            ],
        ),
        html.Div(className='eleven columns', children=[
            trajectory_graph := dcc.Graph(
                id='trajectory-graph',
                figure=fbp.effector_trajectories(states_0),
            ),
        ]),
    ]),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            unit_samples_graph := dcc.Graph(
                id='unit-samples-graph',
                figure=fbp.activity_sample_units(
                    states_0.net.hidden,
                    n_samples=5,
                    key=key_eval,
                ),
            ),
        ]),
        html.Div(className='six columns', children=[
            units_heatmap_graph := dcc.Graph(
                id='units-heatmap-graph',
                figure=fbp.activity_heatmap(states_0.net.hidden[0]),
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
        ]),
    ])
])

@callback(
    Output(trajectory_graph, 'figure'),
    Output(unit_samples_graph, 'figure'),
    Output(units_heatmap_graph, 'figure'),
    Output('trial--slider', 'value'),
    # Output('unselect--button', 'n_clicks'),
    Input(trajectory_graph, 'figure'),
    Input(unit_samples_graph, 'figure'),
    Input(units_heatmap_graph, 'figure'),
    Input(trajectory_graph, 'clickData'),
    Input(unit_samples_graph, 'clickData'),
    Input('replicate--slider', 'value'),
    Input('trial--slider', 'value'),
    Input('unselect--button', 'n_clicks'),
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
):
    states_i = tree_take(states, replicate - 1)

    fig_traj = fbp.effector_trajectories(states_i)
    fig_unit_samples = fbp.activity_sample_units(
        states_i.net.hidden,
        n_samples=5,
        key=key_eval,
    )

    if ctx.triggered_id == 'unselect--button':
        fig_traj.update_traces(opacity=1)
        fig_unit_samples.update_traces(opacity=1)

    # Select individual trials.
    if "clickData" in ctx.triggered[0]['prop_id']:
        curve_number = ctx.triggered[0]['value']['points'][0]['curveNumber']
        fig_clicked = {
            'trajectory-graph': fig_traj,
            'unit-samples-graph': fig_unit_samples,
        }[ctx.triggered_id]
        name = fig_clicked.data[curve_number].name
        fig_traj.update_traces(
            selectedpoints=[],
            selector=lambda trace: trace.name != name,
        )
        fig_unit_samples.update_traces(
            opacity=0.3,
            selectedpoints=[],
            selector=lambda trace: trace.name != name,
        )
        fig_unit_samples.update_traces(
            line_width=3,
            opacity=1,
            selector=lambda trace: trace.name == name,
        )

        i_trial = int(name)

    fig_units_heatmap = fbp.activity_heatmap(states_0.net.hidden[i_trial])

    return fig_traj, fig_unit_samples, fig_units_heatmap, i_trial


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
