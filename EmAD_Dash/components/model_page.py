
from components.data_page import *

''' Defining the Model training interface '''
''' A- Define Train model card '''
label_width = 5
input_width = 6

select_model = html.Div([
    dcc.Dropdown(
        id='select-model',
        options=[
            {'label': 'Isolation Forest', 'value': 'iforest'},
            {'label': 'K Nearest Neighbors', 'value': 'knn'},
            {'label': 'Local Outlier Factor (LOF)', 'value': 'lof'}
        ],
        value='iforest'
    ),
])


model_feature = dbc.FormGroup(
    [dbc.Label("", html_for="offset_samples", width=label_width,id='model_feature_label'),
    dbc.Col(html.Div(id='model_feature_content'),width=input_width,className="m-1"),],
    row=True,)

model_contamination_form = dbc.FormGroup(
    [dbc.Label("Contamination:", html_for="model_contamination", width=label_width,),
    dbc.Col(dcc.Slider(id='model_contamination',min=0.001,max=0.2,
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

train_button_form = dbc.FormGroup([
dbc.Button("Train", id="train_btn",  color="success",block=True, className="mx-2"),
],
row=True)

test_train_button_form = dbc.FormGroup([
dbc.Button("Test on Training data", id="test_train_btn",  color="success",block=True, className="mx-2"),
],
row=True)

alert_model_trained = dbc.Alert(
            html.H5("Model Has been Trained!"),
            id="alert_model_trained",
            is_open=False,
            duration=5000,
        )

train_form = dbc.Form([model_feature, model_contamination_form, train_button_form, test_train_button_form])
train_card=dbc.Card([
alert_model_trained,
html.H5("Model Training", className="text-primary mx-auto"),
html.Hr(),
select_model,
html.Hr(),
train_form,
],body=True,
className="mt-3")

''' A- Define Train model card '''

''' B- Generate the train Graphs Card'''
train_graphs_div=html.Div([dbc.Tabs(
        [
            dbc.Tab(label="Scatter", tab_id="train_scatter"),
            dbc.Tab(label="Line Plots", tab_id="train_line"),
        ],
        id="train-graph-tabs",
        active_tab="train_scatter",
        ),
        html.Div(id="train-graph-tab-content", className="p-0"),
        ],
        className="mt-3"),

''' B- Generate the trained Graphs Card'''


''' C- Define test model card '''
alert_model_tested = dbc.Alert(
            html.H5("Data Loaded Successfuly!"),
            id="alert_model_tested",
            is_open=False,
            duration=5000,
        )

upload_model = dcc.Upload(id='upload-model',
    children=html.Div(id="upload_model_text",children=['Drag and Drop or ',html.A('Select Model File')]),
    style=UPLOAD_STYLE,
    multiple=False,
    className="btn btn-outline-primary mx-auto"
)

test_switches = dbc.Checklist(
options=[{"label": "Calculate ROC", "value": 1},
{"label": "Calculate AUC", "value": 2},
{"label": "Has Time stamp field", "value": 3},
],
value=[1],
id="test-switches",
switch=True,)


test_card=dbc.Card([
alert_data_loaded,
html.H5("Test a Trained Model", className="text-primary mx-auto"),
html.Hr(),
upload_model,
html.Hr(),
load_switches,
dbc.Button("Test Model", id="test",  color="success",block=True, className="mx-2 mt-4")
],body=True,
className="mt-3")

''' C- Define the Load data card '''

''' Defining the Model training and testing interface '''

model_page_container = html.Div(
    [
        # dcc.Store(id="store"),
        html.H1("2- Model Training and Testing"),
        dbc.Card([
        dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Train", tab_id="train",),
                        dbc.Tab(label="Test", tab_id="test"),
                    ],
                    id="model-tabs",
                    active_tab="train",
                    card=True
                )
        ), dbc.CardBody(
        html.Div(id="model-tab-content", children="Test")
        )
        ])

    ]
)
#
def model_tabs_callbacks(app):
    '''Data Generation card callbacks'''
    @app.callback(
    [Output('model_feature_label', 'children'),
    Output('model_feature_content', 'children')],
    [Input('select-model', 'value')]
    )
    def update_output(value):
        if(value=="iforest"):
            return ("Random State", dbc.Input(id='model_feature_value',type="number", value=3))
        elif (value=="knn"):
            knn_average_chklst=dbc.Checklist(
            options=[{"label": "Average method?", "value": 1},],
            value=1,id="model_feature_value",switch=True,)
            return ("", knn_average_chklst)
        elif (value=="lof"):
            return ("# of Nieghbors", dbc.Input(id='model_feature_value',type="number", value=3))
        else:
            return (False, 2)

    @app.callback(
    [Output('loaded_model_store', 'data'),
    Output('alert_model_trained', 'is_open')],
    [Input('train_btn', 'n_clicks')],
    [State('select-model', 'value'),
    # State('model_feature_value', 'value'), # add ths > , model_feature
    State('model_contamination', 'value')]
    )
    def train_model(n, model_name, contamination):
        if n is None:
            raise PreventUpdate
        else:
            if model_name =='iforest':
                result = train_model_iforest(model_feature, contamination)
                print(result)
                return result, True

            elif model_name =='knn':
                result = train_model_knn(model_feature, contamination)
                return result, True

            elif model_name =='iforest':
                result = train_model_lof(model_feature, contamination)
                return result, True
            return {'trained':False}, False

    '''Data Generation card callbacks'''
#
#     '''Data Load card callbacks'''
#     @app.callback(
#     [Output('loaded_data_store', 'data'),
#     Output('alert-data-loaded', 'is_open')],
#     [Input('load', 'n_clicks')],
#     [State('upload-data', 'contents'),
#     State('upload-data', 'filename'),
#     State('load_link', 'value'),
#     State('load-switches', 'value'),
#     State('split_data', 'value')]
#     )
#     def load_data(n,contents, file_name, file_link,switch_values, training_data_ratio):
#         if n is None:
#             raise PreventUpdate
#         else:
#             # TODO: load data from link
#             data = load_file_to_df(contents, file_name, training_data_ratio)
#
#             return data, True
#
#     '''Data Load card callbacks'''
#
#     @app.callback(
#         Output("data-graph-tab-content", "children"),
#         [Input("data-graph-tabs", "active_tab"),
#         Input("loaded_data_store", "data"),
#         Input("generated_data_store", "data")],
#         # [State("loaded_data_store","data"),
#         # State("generated_data_store","data"),]
#     )
#     def render_tab_graphs(active_tab, loaded,generated):
#
#         loaded = loaded or {'loaded':False}
#         generated = generated or {'loaded':False}
#
#         if (active_tab is not None):
#             if loaded['loaded'] is True:
#                 df = DataStorage.loaded_data['xtr']
#             elif generated['loaded'] is True:
#                 df = DataStorage.loaded_data['xtr']
#             else:
#                 return "No Data to represent!"
#
#             if active_tab == "scatter":
#                 return dcc.Graph(figure=update_scatter_matrix())
#             elif active_tab == "line":
#                 return dcc.Graph(figure=update_line_plots())
#             elif active_tab == "table":
#                 return dt.DataTable(
#                 id='table',
#                 columns=[{"name": i, "id": i} for i in df.columns],
#                 data=df.to_dict('records'),
#                 page_size=15,
#                 style_cell={'textAlign': 'left'},
#                 # fixed_rows={'headers': True},
#                 )
#
#         return "No tab selected"
#
    '''Data main tabs control'''
    @app.callback(
        Output("model-tab-content", "children"),
        [Input("model-tabs", "active_tab")]
    )
    def render_tab_content(active_tab):
        if active_tab is not None:
            def one_third_two_thirds(one, two):
                content = dbc.Container([
                dbc.Row([
                    dbc.Col(one, width=4),
                    dbc.Col(two, width=8),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Button("1- Data Preparation",href="/page-1", id='btn_to_data', className="mx-auto p-2 btn-secondary", block=True,),
                     width=4, className="ml-auto mt-4"),
                    dbc.Col(dbc.Button(href="/page-3",disabled=True, id='btn_to_deployment', className="mx-auto p-2 btn-success", block=True,),
                     width=4, className="mr-auto mt-4"),
                ])
                ])
                return content

            if active_tab == "train":
                return one_third_two_thirds(train_card, html.Div(id="train_fig"))

            elif active_tab == "load":
                return one_third_two_thirds(test_card, data_graphs_div)


        return "No tab selected"
#
#     @app.callback(
#         Output("upload_box_text", "children"),
#         [Input("upload_box", "contents")]
#     )
#     def display_file_name(filename):
#         print("called")
#         return html.h6(filename)

    @app.callback(
        [Output("btn_to_deployment", "children"),
        Output("btn_to_deployment", "disabled")],
        [Input("trained_model_store", "data")]
    ) 
    def activate_to_train_btn(data):
        if (data is not None):
            return "3- Deployment" , False
        
        return "Train a model to deploy!", True
