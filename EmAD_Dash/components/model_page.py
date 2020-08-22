
from components.data_page import *


label_width = 5
input_width = 6





select_model = html.Div([
    dcc.Dropdown(
        id='select-model',
        options=[
            {'label': 'Principal component analysis(PCA) ', 'value': 'pca'},
            {'label': 'Minimum Covariance Determinant(MCD)', 'value': 'mcd'},
            {'label': 'one-class SVM (OCSVM)', 'value': 'ocsvm'},
            {'label': 'LMDD', 'value': 'lmdd'},
            {'label': 'Local Outlier Factor (LOF)', 'value': 'lof'},
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

# alert_model_trained = dbc.Alert(
#             html.H5("Model Has been Trained!"),
#             id="alert_model_trained",
#             is_open=False,
#             duration=5000,
#         )

model_form = html.Div(id='model_form',)
train_card=dbc.Card([
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
        html.H1("2- Model Training"),
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
    [dbc.Label("Contamination:", html_for="model_contamination", width=label_width,className='ml-2 pl-2'),
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


# 2- MCD
def generate_mcd_panel():
# pyod.models.mcd.MCD(contamination=0.1, store_precision=True, assume_centered=False, support_fraction=None, random_state=None)

    add_mcd_btn = dbc.Button("Add Model", id="add_mcd_btn",  color="success",block=True, className="m-2")
    mcd_panels = dbc.Form([model_contamination_form,add_mcd_btn])
    return mcd_panels


# 3- OCSVM
def generate_ocsvm_panel():
# pyod.models.ocsvm.OCSVM(kernel='rbf', degree=3, shrinking=True, contamination=0.1)
    ocsvm_radio = dbc.FormGroup([
        dbc.Label("Kernel:"),
        dbc.RadioItems(
            id="ocsvm_radio",
            options=[
                {"label": "RBF", "value": 'rbf'},
                {"label": "Ploynomial", "value": 'poly'},
                {"label": "Linear", "value": 'linear'},
                {"label": "Sigmoid", "value": 'sigmoid'},
            ],
            value='rbf',
            className="px-3",
            inline=True,
        ),])

    ocsvm_degree_input = dbc.FormGroup(
    [dbc.Label("Polynomial Degree:", html_for="ocsvm_degree_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="ocsvm_degree_input", value=2, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    ocsvm_switch = dbc.Checklist(
        id="ocsvm_switch",
        options=[{"label": "Shrinking", "value": 1}],
        value=[1],
        switch=True,)

    add_ocsvm_btn = dbc.Button("Add Model", id="add_ocsvm_btn",  color="success",block=True, className="m-2")
    
    ocsvm_panels = dbc.Form([ocsvm_radio,ocsvm_degree_input,ocsvm_switch,model_contamination_form,add_ocsvm_btn])

    return ocsvm_panels

# 4- LMDD
def generate_lmdd_panel():
# pyod.models.lmdd.LMDD(contamination=0.1, n_iter=50, dis_measure='aad', random_state=None)
    lmdd_iteration_input = dbc.FormGroup(
    [dbc.Label("Iterations:", html_for="lmdd_iteration_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="lmdd_iteration_input", value=30, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    lmdd_radio = dbc.FormGroup([
        dbc.Label("Kernel:"),
        dbc.RadioItems(
            id="lmdd_radio",
            options=[
                {"label": "Average Absolute Deviation", "value": 'aad'},
                {"label": "Variance", "value": 'var'},
                {"label": "Interquartile Range", "value": 'iqr'},
            ],
            value='aad',
            className="px-3",
            # inline=True,
        ),])    

    add_lmdd_btn = dbc.Button("Add Model", id="add_lmdd_btn",  color="success",block=True, className="m-2")
    lmdd_panels = dbc.Form([lmdd_iteration_input,lmdd_radio,model_contamination_form,add_lmdd_btn])
    return lmdd_panels

# 5- LOF
def generate_lof_panel():
# pyod.models.lof.LOF(n_neighbors=20, algorithm='auto', leaf_size=30, contamination=0.1)
    lof_neighbers_input = dbc.FormGroup(
    [dbc.Label("Neighbers:", html_for="lof_neighbers_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="lof_neighbers_input", value=20, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    lof_leaf_input = dbc.FormGroup(
    [dbc.Label("Leaf Size:", html_for="lof_leaf_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="lof_leaf_input", value=30, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    lof_radio = dbc.FormGroup([
        dbc.Label("Algorithm:"),
        dbc.RadioItems(
            id="lof_radio",
            options=[
                {"label": "Auto", "value": 'auto'},
                {"label": "BallTree", "value": 'ball_tree'},
                {"label": "KDTree", "value": 'kd_tree'},
                {"label": "Brute-force", "value": 'brute'},
            ],
            value='auto',
            className="px-3",
            # inline=True,
        ),])    

    add_lof_btn = dbc.Button("Add Model", id="add_lof_btn",  color="success",block=True, className="m-2")
    lof_panels = dbc.Form([lof_neighbers_input, lof_leaf_input,lof_radio,model_contamination_form,add_lof_btn])
    return lof_panels


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

        if(value=="ocsvm"):
            ocsvm_panel = generate_ocsvm_panel()
            return ocsvm_panel

        if(value=="lmdd"):
            lmdd_panel = generate_lmdd_panel()
            return lmdd_panel

        if(value=="lof"):
            lmdd_panel = generate_lof_panel()
            return lmdd_panel


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
                    dbc.Col(dbc.Button("Train models before testing",href="/page-3",disabled=True, id='btn_to_testing', className="mx-auto p-2 btn-success", block=True,),
                     width=4, className="mr-auto mt-4"),
                ])
                ], className='px-0 mx-0')
                return content

            if active_tab == "train":
                return one_third_two_thirds(train_card, added_models_card)

        return "No tab selected"

    # 0- Update Table Display
    @app.callback(
        Output("added_models_table", "children"),
        # [Input({'type': 'table_signal', 'index': ALL}, 'data')]
        [Input('pca_add_signal', 'data'),
        Input('mcd_add_signal', 'data'),
        Input('ocsvm_add_signal', 'data'),
        Input('lmdd_add_signal', 'data'),
        Input('lof_add_signal', 'data'),
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

    # 3- OCSVM Callback
    @app.callback( #ocsvm_radio,ocsvm_degree_input,ocsvm_switch,model_contamination_form,add_ocsvm_btn
        Output('ocsvm_add_signal', 'data'),
        [Input("add_ocsvm_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("ocsvm_radio", "value"),
        State("ocsvm_degree_input", "value"),
        State("ocsvm_switch", "value"),]
    ) 
    def add_ocsvm_clbk(n,contamination,radio,degree,switch):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.ocsvm import OCSVM
            clf = OCSVM(contamination=contamination, kernel=radio,degree=degree,shrinking=(1 in switch))
            name = 'OCSVM ({},{},{},{})'.format(contamination,radio,degree, 1 in switch)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}


    # 4- LMDD Callback
    @app.callback( #lmdd_iteration_input,lmdd_radio,model_contamination_form,add_lmdd_btn
        Output('lmdd_add_signal', 'data'),
        [Input("add_lmdd_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("lmdd_radio", "value"),
        State("lmdd_iteration_input", "value")]
    ) 
    def add_lmdd_clbk(n,contamination,radio,iteration):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.lmdd import LMDD
            clf = LMDD(contamination=contamination, n_iter=iteration, dis_measure=radio, random_state=random_state)
            name = 'LMDD ({},{},{},{})'.format(contamination,radio,iteration,radio)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}


    # 5- LOF Callback
    @app.callback( #lof_neighbers_input, lof_leaf_input,lof_radio,model_contamination_form,add_lof_btn
        Output('lof_add_signal', 'data'),
        [Input("add_lof_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("lof_neighbers_input", "value"),
        State("lof_leaf_input", "value"),
        State("lof_radio", "value")]
    ) 
    def add_lof_clbk(n,contamination,neighbers,leaf,algorithm):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.lof import LOF
            clf = LOF(contamination=contamination, n_neighbors=neighbers, leaf_size=leaf, algorithm=algorithm)
            name = 'LOF ({},{},{},{})'.format(contamination,neighbers,leaf,algorithm)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}


    @app.callback(
    [Output('loading', 'children'),
    Output('train_signal', 'data'),
    Output('btn_to_testing', 'disabled'),
    Output('btn_to_testing', 'children')],
    [Input('train_btn', 'n_clicks')]
    )
    def train_added_models(n):
        if (n is None):
            raise PreventUpdate
        else:
            train_models()
            return '', {'added': True}, False, '3- Model Testing'