
from components.data_page import *


label_width = 5
input_width = 6

dcc.Store(id={'type': 'table_signal','index': 'pca'})
dcc.Store(id={'type': 'table_signal','index': 'train'})




select_model = html.Div([
    dcc.Dropdown(
        id='select-model',
        options=[
            {'label': 'Principal component analysis (PCA) ', 'value': 'pca'},
            {'label': 'Minimum Covariance Determinant (MCD)', 'value': 'mcd'},
        ],
        value=''
    ),
])



# Note to delete below:
model_feature = dbc.FormGroup(
    [dbc.Label("", html_for="offset_samples", width=label_width,id='model_feature_label'),
    dbc.Col(html.Div(id='model_parameters'),width=input_width,className="m-1"),],
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

train_btn = dbc.Button("Train", id="train_btn",  color="success",block=True, className="m-2")

alert_model_trained = dbc.Alert(
            html.H5("Model Has been Trained!"),
            id="alert_model_trained",
            is_open=False,
            duration=5000,
        )

model_form = html.Div(id='model_form',)
train_card=dbc.Card([
alert_model_trained,
html.H5("Model Training", className="text-primary mx-auto"),
html.Hr(),
select_model,
html.Hr(),
model_form,
],body=True,
className="m-3")

added_models_card=dbc.Card([
html.H5("Added Models", className="text-primary mx-auto"),
html.Hr(),
html.Div(id='added_models_table')
],body=True,
className="my-3")


''' Defining the Model training and testing interface '''

model_page_container = html.Div(
    [
        html.H1("2- Model Training and Testing"),
        dbc.Card([
        dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Train", tab_id="train",),
                        # dbc.Tab(label="Test", tab_id="test"),
                    ],
                    id="model-tabs",
                    active_tab="train",
                    card=True
                )
        ), dbc.CardBody(
        html.Div(id="model-tab-content", children="Test", className='px-0 mx-0')
        )
        ])

    ]
)




''' Defining Models Interfaces'''
model_feature = dbc.FormGroup(
    [dbc.Label("", html_for="offset_samples", width=label_width,id='model_feature_label'),
    dbc.Col(html.Div(id='model_parameters'),width=input_width,className="m-1"),],
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


# 1- PCA
def generate_pca_panel():
# pyod.models.pca.PCA(contamination=0.1, svd_solver='auto', weighted=True, standardization=True, random_state=None)    
    pca_radio = dbc.FormGroup([
        dbc.Label("SVD Solver:"),
        dbc.RadioItems(
            id="pca_radio",
            options=[
                {"label": "Auto", "value": 'auto'},
                {"label": "Full", "value": 'full'},
                {"label": "Arpack", "value": 'arpack'},
                {"label": "Randomized", "value": 'randomized'},
            ],
            value='auto',
            className="px-3",
            inline=True,
        ),])
    pca_switches = dbc.Checklist(
        id="pca_switches",
        options=[{"label": "Weighted", "value": 1},
        {"label": "Standarize", "value": 2},],
        value=[1,2],
        switch=True,)

    add_pca_btn = dbc.Button("Add Model", id="add_pca_btn",  color="success",block=True, className="m-2")
    
    pca_panels = dbc.Form([pca_radio,pca_switches,model_contamination_form,add_pca_btn])

    return pca_panels


# 1- MCD
def generate_mcd_panel():
# pyod.models.mcd.MCD(contamination=0.1, store_precision=True, assume_centered=False, support_fraction=None, random_state=None)

    add_mcd_btn = dbc.Button("Add Model", id="add_mcd_btn",  color="success",block=True, className="m-2")
    mcd_panels = dbc.Form([model_contamination_form,add_mcd_btn])
    return mcd_panels

''' Defining Models Interfaces'''


def model_tabs_callbacks(app):
    '''Data Generation card callbacks'''
    @app.callback(
    Output('model_form', 'children'),
    [Input('select-model', 'value')]
    )
    def update_output(value):
        if(value=="pca"):
            pca_panel = generate_pca_panel()
            return pca_panel

        if(value=="mcd"):
            mcd_panel = generate_mcd_panel()
            return mcd_panel

        else:
            return html.H4('Select Model to Add', className='mx-auto')



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
                    dbc.Col(dbc.Button("3- Model Testing",href="/page-3",disabled=True, id='btn_to_testing', className="mx-auto p-2 btn-success", block=True,),
                     width=4, className="mr-auto mt-4"),
                ])
                ], className='px-0 mx-0')
                return content

            if active_tab == "train":
                return one_third_two_thirds(train_card, added_models_card)

        return "No tab selected"


    @app.callback(
        Output("added_models_table", "children"),
        # [Input({'type': 'table_signal', 'index': ALL}, 'data')]
        [Input('pca_add_signal', 'data'),
        Input('mcd_add_signal', 'data'),
        Input('train_signal', 'data')]
    ) 
    def update_added_models_table(*args):
        if (len(DataStorage.model_list)>0):
            added_models = html.Div([
                dbc.Row(generate_model_table()),
                dbc.Row(dbc.Col(dbc.Spinner(html.Div(id='loading')), width=12)),
                dbc.Row(train_btn),
            ])
            return added_models
        
        return html.H4('No Models Added')

    # 1- PCA Callback
    @app.callback(
        Output('pca_add_signal', 'data'),
        [Input("add_pca_btn", "n_clicks")],
        [State("pca_radio", "value"),
        State("pca_switches", "value"),
        State("model_contamination", "value"),]
    ) 
    def add_pca_clbk(n, solver, switches,contamination):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.pca import PCA
            clf = PCA(contamination=contamination, svd_solver=solver, weighted=(1 in switches),
             standardization=(2 in switches), random_state=random_state)
            name = 'PCA ({},{},{},{})'.format(contamination,solver,(1 in switches),(2 in switches))
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 2- MCD Callback
    @app.callback(
        Output('mcd_add_signal', 'data'),
        [Input("add_mcd_btn", "n_clicks")],
        [State("model_contamination", "value"),]
    ) 
    def add_mcd_clbk(n,contamination):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.mcd import MCD
            clf = MCD(contamination=contamination, random_state=random_state)
            name = 'MCD ({})'.format(contamination)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}


    @app.callback(
    [Output('loading', 'children'),
    Output('train_signal', 'data'),
    Output('btn_to_testing', 'disabled')],
    [Input('train_btn', 'n_clicks')]
    )
    def train_added_models(n):
        if (n is None):
            raise PreventUpdate
        else:
            train_models()
            return '', {'added': True}, False