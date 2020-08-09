import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

''' Defining the Data generator interface '''

''' A- Define the Generate data card '''
select_generator = html.Div([
    dcc.Dropdown(
        id='select-generator',
        options=[
            {'label': 'PyOD Simple', 'value': 'simple'},
            {'label': 'PyOD Clusters', 'value': 'clusters'}
        ],
        value='simple'
    ),
])

n_train_form = dbc.FormGroup(
    [dbc.Label("Training Samples:", html_for="training_samples", width=6,),
    dbc.Col(dbc.Input(type="number", id="n_train", value=1000, step=100,),width=4,className="m-1"),],
    row=True)

n_test_form = dbc.FormGroup(
    [dbc.Label("Testing Samples:", html_for="testing_samples", width=6,),
    dbc.Col(dbc.Input(type="number", id="n_test", value=500, step=100,),width=4,className="m-1"),],
    row=True,)

n_features_form = dbc.FormGroup(
    [dbc.Label("Features:", html_for="features_samples", width=6,),
    dbc.Col(dbc.Input(type="number", id="n_features", value=2, step=1,),width=4,className="m-1"),],
    row=True,)

n_clusters_form = dbc.FormGroup(
    [dbc.Label("Clusters:", html_for="clusters_samples", width=6,),
    dbc.Col(dbc.Input(type="number", id="n_clusters", value=2, step=1,),width=4,className="m-1"),],
    row=True,)

offset_form = dbc.FormGroup(
    [dbc.Label("Offset:", html_for="offset_samples", width=6,),
    dbc.Col(dbc.Input(type="number", id="offset", value=10, step=1,),width=4,className="m-1"),],
    row=True,)

contamination_form = dbc.FormGroup(
    [dbc.Label("Contamination:", html_for="contamination_samples", width=4,),
    dbc.Col(dcc.Slider(id='contamination',min=0.001,max=0.2,
    step=0.01,
    marks={
    0.001: '0.001',
    0.1: '0.1',
    0.2: '0.2'
    },
    value=0.1,
    tooltip={'placement':'top'}
    )
    ,width=7,
    className="m-1"),],
    row=True,)

button_form = dbc.FormGroup([
dbc.Button("Generate", id="generate",  color="success",block=True, className="mx-2"),
],
row=True)

alert_data_generated = dbc.Alert(
            html.H5("Data Generated!"),
            id="alert-data-generated",
            is_open=False,
            duration=5000,
        )

generate_form = dbc.Form([n_train_form, n_test_form, n_features_form, n_clusters_form, offset_form, contamination_form, button_form])
generate_card=dbc.Card([
alert_data_generated,
select_generator,
html.Hr(),
generate_form,
],body=True,
className="m-3")

generate_container = dbc.Row(generate_card,form=True, className="col-6 mx-auto")
''' A- Define the Generate data card '''





''' Defining the Data generator interface '''

taps_with_graphs = dbc.Container(
    [
        # dcc.Store(id="store"),
        html.H1("1- Data Preparation"),

        dbc.Tabs(
            [
                dbc.Tab(label="Generate", tab_id="generate"),
                dbc.Tab(label="Load", tab_id="load"),
            ],
            id="data-tabs",
            active_tab="generate",
        ),

        html.Div(id="data-tab-content", children="Test"),
        # html.Hr(),
        dbc.Card([
        dbc.Tabs(
            [
                dbc.Tab(label="Scatter", tab_id="scatter"),
                dbc.Tab(label="Histograms", tab_id="histogram"),
            ],
            id="data-graph-tabs",
            active_tab="scatter",
        ),
        html.Div(id="data-graph-tab-content", className="p-4"),
        ],body=True),

    ]
)

def data_tabs_callbacks(app):
    '''Data Generation card callbacks'''
    @app.callback(
    [Output('n_clusters', 'disabled'),
    Output('n_clusters', 'value')],
    [Input('select-generator', 'value')]
    )
    def update_output(value):
        if(value=="simple"):
            return (True, 1)
        else:
            return (False, 2)

    @app.callback(
    [Output('data_store', 'data'),
    Output('alert-data-generated', 'is_open')],
    [Input('generate', 'n_clicks')],
    [State('n_train', 'value'),
    State('n_test', 'value'),
    State('n_features', 'value'),
    State('n_clusters', 'value'),
    State('offset', 'value'),
    State('contamination', 'value')]
    )
    def generate_data(n,n_train, n_test, n_features,n_clusters,offset,contamination):
        if n is None:
            return {'temp':0}, False
        else:
            return {'temp':0}, True

    '''Data Generation card callbacks'''

    @app.callback(
        Output("data-graph-tab-content", "children"),
        [Input("data-graph-tabs", "active_tab")]
    )
    def render_tab_content(active_tab):

        # generate 100 multivariate normal samples
        data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)

        scatter = go.Figure(
            data=[go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers")]
        )
        hist_1 = go.Figure(data=[go.Histogram(x=data[:, 0])])
        hist_2 = go.Figure(data=[go.Histogram(x=data[:, 1])])

        if active_tab is not None:
            if active_tab == "scatter":
                dbc.Card
                return dcc.Graph(figure=scatter)
            elif active_tab == "histogram":
                return dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=hist_1), width=6),
                        dbc.Col(dcc.Graph(figure=hist_2), width=6),
                    ]
                )
        return "No tab selected"

    @app.callback(
        Output("data-tab-content", "children"),
        [Input("data-tabs", "active_tab")]
    )
    def render_tab_content(active_tab):

        if active_tab is not None:
            if active_tab == "generate":
                return generate_container
            elif active_tab == "load":
                return "Load"
        return "No tab selected"
