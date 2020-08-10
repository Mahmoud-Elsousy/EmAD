import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

from components.data_functions import generate_data_simple, generate_data_clusters

''' Defining the Data generator interface '''
dcc.Store(id="refresh_figures")
''' A- Define the Generate data card '''
label_width = 5
input_width = 6
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
    [dbc.Label("Training Samples:", html_for="training_samples", width=label_width,),
    dbc.Col(dbc.Input(type="number", id="n_train", value=1000, step=100,),width=input_width,className="m-1"),],
    row=True)

n_test_form = dbc.FormGroup(
    [dbc.Label("Testing Samples:", html_for="testing_samples", width=label_width,),
    dbc.Col(dbc.Input(type="number", id="n_test", value=500, step=100,),width=input_width,className="m-1"),],
    row=True,)

n_features_form = dbc.FormGroup(
    [dbc.Label("Features:", html_for="features_samples", width=label_width,),
    dbc.Col(dbc.Input(type="number", id="n_features", value=2, step=1,),width=input_width,className="m-1"),],
    row=True,)

n_clusters_form = dbc.FormGroup(
    [dbc.Label("Clusters:", html_for="clusters_samples", width=label_width,),
    dbc.Col(dbc.Input(type="number", id="n_clusters", value=2, step=1,),width=input_width,className="m-1"),],
    row=True,)

offset_form = dbc.FormGroup(
    [dbc.Label("Offset:", html_for="offset_samples", width=label_width,),
    dbc.Col(dbc.Input(type="number", id="offset", value=10, step=1,),width=input_width,className="m-1"),],
    row=True,)

contamination_form = dbc.FormGroup(
    [dbc.Label("Contamination:", html_for="contamination_samples", width=label_width,),
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
    ,width=input_width,
    className="t-1"),],
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
html.H5("Generate Data", className="text-primary mx-auto"),
html.Hr(),
select_generator,
html.Hr(),
generate_form,
],body=True,
className="mt-3")

''' A- Define the Generate data card '''

''' B- Generate the Graphs Card'''
data_graphs_div=html.Div([dbc.Tabs(
        [
            dbc.Tab(label="Scatter", tab_id="scatter"),
            dbc.Tab(label="Histograms", tab_id="histogram"),
            dbc.Tab(label="Table", tab_id="table"),
        ],
        id="data-graph-tabs",
        active_tab="scatter",
        ),
        html.Div(id="data-graph-tab-content", className="p-0"),
        ],#body=True,
        className="mt-3"),

''' B- Generate the Graphs Card'''


''' C- Define the Load data card '''
UPLOAD_STYLE= {
    'width': '100%',
    'height': '70px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px'
}


alert_data_loaded = dbc.Alert(
            html.H5("Data Loaded Successfuly!"),
            id="alert-data-loaded",
            is_open=False,
            duration=5000,
        )


radios_load = dbc.FormGroup(
    [dbc.Label("Source", html_for="load_radio", width=label_width-2),
    dbc.Col(
        dbc.RadioItems(
            id="load_radio",
            options=[{"label": "Local File", "value": 1},
                {"label": "Website", "value": 2},],
        ),
        width=input_width+2,),
    ],row=True,
)

upload_box = dcc.Upload(id='upload-data',
    children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
    style=UPLOAD_STYLE,
    multiple=True,
    className="btn btn-outline-primary mx-auto"
)

load_link_form = dbc.FormGroup(
    [dbc.Label("Link:", html_for="load_link", width=label_width-2,
    className="pl-4 pt-3", ),
    dbc.Col(dbc.Input(type="url", id="load_link",placeholder="https://github..."),width=input_width+2,
    className="pt-3"),],
    row=True,)

load_switches = dbc.Checklist(
options=[{"label": "Check for NaN", "value": 1},
{"label": "Generate Header", "value": 2},
{"label": "Has Time stamp field", "value": 3},
{"label": "Shuffel Data", "value": 4},
],
value=[1],
id="load-switches",
switch=True,)

split_form = html.Div(dbc.FormGroup(
    [dbc.Label("Training %:", html_for="split_data", width=label_width-1,className="pl-4"),
    dbc.Col(dcc.Slider(id='split_data',min=0,max=100,
    step=1,
    marks={
    0: '0',
    50: '50',
    100: '100'
    },
    value=70,
    tooltip={'placement':'top'}
    )
    ,width=input_width+1,
    className="pt-3"),],
    row=True,),className="pt-3")

load_card=dbc.Card([
alert_data_loaded,
html.H5("Load Data from file", className="text-primary mx-auto"),
html.Hr(),
upload_box,
# html.Hr(),
load_link_form,
html.Hr(),
load_switches,
split_form,
dbc.Button("Load File", id="load",  color="success",block=True, className="mx-2 mt-4")
],body=True,
className="mt-3")

''' C- Define the Load data card '''

''' D- Define the Upload functions '''

''' D- Define the Upload functions '''


''' Defining the Data generator interface '''

taps_with_graphs = html.Div(
    [
        # dcc.Store(id="store"),
        html.H1("1- Data Preparation"),
        dbc.Card([
        dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Generate", tab_id="generate",),
                        dbc.Tab(label="Load", tab_id="load"),
                    ],
                    id="data-tabs",
                    active_tab="load",
                    card=True
                )
        ), dbc.CardBody(
        html.Div(id="data-tab-content", children="Test")
        )
        ])

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
    [Output('generated_data_store', 'data'),
    Output('alert-data-generated', 'is_open')],
    [Input('generate', 'n_clicks')],
    [State('select-generator', 'value'),
    State('n_train', 'value'),
    State('n_test', 'value'),
    State('n_features', 'value'),
    State('n_clusters', 'value'),
    State('offset', 'value'),
    State('contamination', 'value')]
    )
    def generate_data(n, generator, n_train, n_test, n_features,n_clusters,offset,contamination):
        if n is None:
            raise PreventUpdate
        else:
            if generator =='simple':
                data = generate_data_simple(n_train, n_test, n_features, contamination,offset)
                return data, True

            elif generator == 'clusters':
                data = generate_data_clusters(n_train, n_test, n_clusters, n_features, contamination,offset)
                return data, True

            return {'loaded':False}, True

    '''Data Generation card callbacks'''

    '''Data Load card callbacks'''
    @app.callback(
    [Output('loaded_data_store', 'data'),
    Output('alert-data-loaded', 'is_open')],
    [Input('load', 'n_clicks')],
    [State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('load_link', 'value'),
    State('load-switches', 'value'),
    State('split_data', 'value')]
    )
    def load_data(n,contents, file_name, file_link,switch_values, training_data_ratio):
        if n is None:
            raise PreventUpdate
        else:
            # TODO: load data from link
            data = load_file_to_dict(contents, filename, training_data_ratio)

            return data, True

    '''Data Load card callbacks'''

    @app.callback(
        Output("data-graph-tab-content", "children"),
        [Input("data-graph-tabs", "active_tab")],
        [State("loaded_data_store","data"),
        State("generated_data_store","data"),]
    )
    def render_tab_graphs(active_tab, loaded,generated):

        # generate 100 multivariate normal samples
        data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)

        scatter = go.Figure(
            data=[go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers")]
        )
        hist_1 = go.Figure(data=[go.Histogram(x=data[:, 0])])
        hist_2 = go.Figure(data=[go.Histogram(x=data[:, 1])])

        loaded = loaded or {'loaded':False}
        generated = generated or {'loaded':False}

        if (active_tab is not None):
            if loaded['loaded']==True:
                df = pd.DataFrame.from_dict(loaded['xtr'])
                print(loaded['xtr'])
            elif generated['loaded']==True:
                df = pd.DataFrame.from_dict(generated['xtr'])
            else:
                return "No Data to represent!"

            if active_tab == "scatter":
                return dcc.Graph(figure=scatter)
            elif active_tab == "histogram":
                return dcc.Graph(figure=hist_1)
            elif active_tab == "table":

                return dt.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),)

        return "No tab selected"

    '''Data main tabs control'''
    @app.callback(
        Output("data-tab-content", "children"),
        [Input("data-tabs", "active_tab")]
    )
    def render_tab_content(active_tab):
        if active_tab is not None:
            if active_tab == "generate":
                return dbc.Row([
                dbc.Col(generate_card, width=4),
                dbc.Col(data_graphs_div, width=8),
                ])
            elif active_tab == "load":
                return dbc.Row([
                dbc.Col(load_card, width=4),
                dbc.Col(data_graphs_div, width=8),
                ])

        return "No tab selected"
