
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
            {'label': 'Linear Method for Deviation-based OD(LMDD)', 'value': 'lmdd'},
            {'label': 'Local Outlier Factor(LOF)', 'value': 'lof'},
            {'label': 'Connectivity-Based Outlier Factor(COF)', 'value': 'cof'},
            {'label': 'Clustering Based Local Outlier Factor(CBLOF)', 'value': 'cblof'},
            {'label': 'Histogram- based outlier detection(HBOS)', 'value': 'hbos'},
            {'label': 'k-Nearest Neighbors(KNN)', 'value': 'knn'},
            {'label': 'Angle-base Outlier Detection(ABOD)', 'value': 'abod'},
            {'label': 'Isolation Forest OD(IForest)', 'value': 'iforest'},
            {'label': 'Feature bagging detector(FB)', 'value': 'fb'},
            {'label': 'Subspace outlier detection(SOD) ', 'value': 'sod'},
        ],
        value=''
    ),
])


train_btn = dbc.Button("Train", id="train_btn",  color="success",block=True, className="m-2")

save_load = dbc.Row([
    dbc.Col(
        dbc.Button("Save list", id="save_list", size="sm",outline=True,color="success",block=True, className="my-1 pl-1",),
        className="pl-4"),

    dbc.Col(
        dbc.Button("Load List", id="load_list", size="sm",outline=True,color="info",block=True, className="my-1 pr-1",),
        className="pr-4"),

],className="px-0 mx-0")

alert_list_saved = dbc.Alert(
            html.H5("List Saved Successfuly!"),
            id="alert_list_saved",
            is_open=False,
            duration=3000,
        )

model_form = html.Div(id='model_form',)
train_card=dbc.Card([
alert_list_saved,
html.H5("Model Training", className="text-primary mx-auto"),
html.Hr(),
select_model,
html.Hr(),
model_form,
save_load,
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


# 6- COF
def generate_cof_panel():
# pyod.models.lmdd.LMDD(contamination=0.1, n_iter=50, dis_measure='aad', random_state=None)
    cof_neighbers_input = dbc.FormGroup(
    [dbc.Label("Neighbers:", html_for="cof_neighbers_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="cof_neighbers_input", value=20, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    add_cof_btn = dbc.Button("Add Model", id="add_cof_btn",  color="success",block=True, className="m-2")
    cof_panels = dbc.Form([cof_neighbers_input,model_contamination_form,add_cof_btn])
    return cof_panels

# 7- CBLOF
def generate_cblof_panel():
# pyod.models.cblof.CBLOF(n_clusters=8, contamination=0.1, alpha=0.9, beta=5,use_weights=False, random_state=None)
    cblof_clusters_input = dbc.FormGroup(
    [dbc.Label("Clusters:", html_for="cblof_clusters_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="cblof_clusters_input", value=20, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    cblof_alpha_input = dbc.FormGroup(
    [dbc.Label("Alpha:", html_for="cblof_alpha_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="cblof_alpha_input", value=0.9, step=0.1,),width=input_width-2,className="m-1"),],
    row=True)

    cblof_beta_input = dbc.FormGroup(
    [dbc.Label("Beta:", html_for="cblof_beta_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="cblof_beta_input", value=5, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    add_cblof_btn = dbc.Button("Add Model", id="add_cblof_btn",  color="success",block=True, className="m-2")
    cof_panels = dbc.Form([cblof_clusters_input, cblof_alpha_input, cblof_beta_input, model_contamination_form,add_cblof_btn])
    return cof_panels

# 8- HBOS
def generate_hbos_panel():
# pyod.models.hbos.HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.1)
    hbos_bins_input = dbc.FormGroup(
    [dbc.Label("Bins:", html_for="hbos_bins_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="hbos_bins_input", value=10, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    hbos_alpha_input = dbc.FormGroup(
    [dbc.Label("Alpha:", html_for="hbos_alpha_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="hbos_alpha_input", value=0.1, step=0.1,),width=input_width-2,className="m-1"),],
    row=True)

    hbos_tol_input = dbc.FormGroup(
    [dbc.Label("Tolerance:", html_for="hbos_tol_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="hbos_tol_input", value=0.5, step=0.1,),width=input_width-2,className="m-1"),],
    row=True)

    add_hbos_btn = dbc.Button("Add Model", id="add_hbos_btn",  color="success",block=True, className="m-2")
    cof_panels = dbc.Form([hbos_bins_input, hbos_alpha_input,hbos_tol_input, model_contamination_form,add_hbos_btn])
    return cof_panels



# 9- KNN
def generate_knn_panel():
# pyod.models.knn.KNN(contamination=0.1, n_neighbors=5, method='largest', algorithm='auto',metric='minkowski')
    knn_neighbers_input = dbc.FormGroup(
    [dbc.Label("Neighbers:", html_for="knn_neighbers_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="knn_neighbers_input", value=5, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    knn_algorithm_radio = dbc.FormGroup([
        dbc.Label("Algorithm:"),
        dbc.RadioItems(
            id="knn_algorithm_radio",
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

    knn_method_radio = dbc.FormGroup([
        dbc.Label("Method:"),
        dbc.RadioItems(
            id="knn_method_radio",
            options=[
                {"label": "Largest", "value": 'largest'},
                {"label": "Mean", "value": 'mean'},
                {"label": "Median", "value": 'median'},
            ],
            value='largest',
            className="px-3",
            # inline=True,
        ),])

    knn_metric_input = dbc.FormGroup(
    [dbc.Label("Metric:", html_for="knn_metric_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="text", id="knn_metric_input", value='minkowski'),width=input_width-2,className="m-1"),],
    row=True)    

    add_knn_btn = dbc.Button("Add Model", id="add_knn_btn",  color="success",block=True, className="m-2")
    panel = dbc.Form([knn_neighbers_input, knn_algorithm_radio, knn_method_radio, knn_metric_input, model_contamination_form,add_knn_btn])
    return panel

# 10- ABOD
def generate_abod_panel():
# pyod.models.abod.ABOD(contamination=0.1, n_neighbors=5, method='fast')
    abod_neighbers_input = dbc.FormGroup(
    [dbc.Label("Neighbers:", html_for="abod_neighbers_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="abod_neighbers_input", value=5, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    abod_radio = dbc.FormGroup([
        dbc.Label("Algorithm:"),
        dbc.RadioItems(
            id="abod_radio",
            options=[
                {"label": "Fast", "value": 'fast'},
                {"label": "Default", "value": 'default'},
            ],
            value='fast',
            className="px-3",
            # inline=True,
        ),])

    add_abod_btn = dbc.Button("Add Model", id="add_abod_btn",  color="success",block=True, className="m-2")
    panel = dbc.Form([abod_neighbers_input, abod_radio,model_contamination_form,add_abod_btn])
    return panel

# 11- IForest
def generate_iforest_panel():
# pyod.models.iforest.IForest(n_estimators=100, contamination=0.1, bootstrap=False,)
    iforest_estimators_input = dbc.FormGroup(
    [dbc.Label("Estimators:", html_for="iforest_estimators_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="iforest_estimators_input", value=100, step=10,),width=input_width-2,className="m-1"),],
    row=True)

    iforest_switch = dbc.Checklist(options=[{"label": "Bootstrap: ", "value": 1},],value=[],id="iforest_switch",switch=True,)


    add_iforest_btn = dbc.Button("Add Model", id="add_iforest_btn",  color="success",block=True, className="m-2")
    panel = dbc.Form([iforest_estimators_input, iforest_switch,model_contamination_form,add_iforest_btn])
    return panel

# 12- Feature Bagging
def generate_fb_panel():
# pyod.models.iforest.IForest(n_estimators=100, contamination=0.1, bootstrap=False,)
    fb_estimators_input = dbc.FormGroup(
    [dbc.Label("Estimators:", html_for="fb_estimators_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="fb_estimators_input", value=10, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    fb_switch = dbc.Checklist(options=[{"label": "Bootstrap: ", "value": 1},],value=[],id="fb_switch",switch=True,)

    fb_radio = dbc.FormGroup([
        dbc.Label("Combination:"),
        dbc.RadioItems(
            id="fb_radio",
            options=[
                {"label": "Average", "value": 'average'},
                {"label": "Max", "value": 'max'},
            ],
            value='average',
            className="px-3",
            inline=True,
        ),])

    add_fb_btn = dbc.Button("Add Model", id="add_fb_btn",  color="success",block=True, className="m-2")
    panel = dbc.Form([fb_estimators_input, fb_switch, fb_radio, model_contamination_form,add_fb_btn])
    return panel

# 13- SOD
def generate_sod_panel():
# pyod.models.sod.SOD(contamination=0.1, n_neighbors=20, ref_set=10, alpha=0.8)
    sod_neighbers_input = dbc.FormGroup(
    [dbc.Label("Neighbers:", html_for="sod_neighbers_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="sod_neighbers_input", value=20, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    sod_refset_input = dbc.FormGroup(
    [dbc.Label("Reference Set:", html_for="sod_refset_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="sod_refset_input", value=10, step=1,),width=input_width-2,className="m-1"),],
    row=True)

    sod_alpha_input = dbc.FormGroup(
    [dbc.Label("Estimators:", html_for="sod_alpha_input", width=label_width+2,),
    dbc.Col(dbc.Input(type="number", id="sod_alpha_input", value=0.8, step=0.1,),width=input_width-2,className="m-1"),],
    row=True)


    add_sod_btn = dbc.Button("Add Model", id="add_sod_btn",  color="success",block=True, className="m-2")
    panel = dbc.Form([sod_neighbers_input, sod_refset_input, sod_alpha_input, model_contamination_form,add_sod_btn])
    return panel


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
            lof_panel = generate_lof_panel()
            return lof_panel

        if(value=="cof"):
            cof_panel = generate_cof_panel()
            return cof_panel

        if(value=="cblof"):
            panel = generate_cblof_panel()
            return panel

        if(value=="hbos"):
            panel = generate_hbos_panel()
            return panel

        if(value=="knn"):
            panel = generate_knn_panel()
            return panel

        if(value=="abod"):
            panel = generate_abod_panel()
            return panel

        if(value=="iforest"):
            panel = generate_iforest_panel()
            return panel

        if(value=="fb"):
            panel = generate_fb_panel()
            return panel

        if(value=="sod"):
            panel = generate_sod_panel()
            return panel


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
        Input('cof_add_signal', 'data'),
        Input('cblof_add_signal', 'data'),
        Input('hbos_add_signal', 'data'),
        Input('knn_add_signal', 'data'),
        Input('abod_add_signal', 'data'),
        Input('sod_add_signal', 'data'),
        Input('iforest_add_signal', 'data'),
        Input('fb_add_signal', 'data'),
        Input('list_loaded', 'data'),
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

    # 6- COF Callback
    @app.callback( #lmdd_iteration_input,lmdd_radio,model_contamination_form,add_lmdd_btn
        Output('cof_add_signal', 'data'),
        [Input("add_cof_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("cof_neighbers_input", "value")]
    ) 
    def add_cof_clbk(n,contamination,neighbers):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.cof import COF
            clf = COF(contamination=contamination, n_neighbors=neighbers)
            name = 'COF ({},{})'.format(contamination,neighbers)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 7- CBLOF Callback
    @app.callback( #lmdd_iteration_input,lmdd_radio,model_contamination_form,add_lmdd_btn
        Output('cblof_add_signal', 'data'),
        [Input("add_cblof_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("cblof_clusters_input", "value"),
        State("cblof_alpha_input", "value"),
        State("cblof_beta_input", "value")]
    ) 
    def add_cblof_clbk(n,contamination,clusters,alpha,beta):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.cblof import CBLOF
            clf = CBLOF(contamination=contamination, n_clusters=clusters, alpha=alpha,beta=beta,random_state=random_state)
            name = 'CBLOF ({},{},{},{})'.format(contamination,clusters,alpha,beta)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 8- HBOS Callback
    @app.callback(
        Output('hbos_add_signal', 'data'),
        [Input("add_hbos_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("hbos_bins_input", "value"),
        State("hbos_alpha_input", "value"),
        State("hbos_tol_input", "value")]
    ) 
    def add_cblof_clbk(n,contamination,bins,alpha,tol):
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.hbos import HBOS
            clf = HBOS(contamination=contamination, n_bins=bins, alpha=alpha,tol=tol)
            name = 'HBOS ({},{},{},{})'.format(contamination,bins,alpha,tol)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 9- KNN Callback
    @app.callback(
        Output('knn_add_signal', 'data'),
        [Input("add_knn_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("knn_neighbers_input", "value"),
        State("knn_algorithm_radio", "value"),
        State("knn_method_radio", "value"),
        State("knn_metric_input", "value")]
    ) 
    def add_knn_clbk(n,contamination,neighbers,algorithm,method,metric):#[knn_neighbers_input, knn_algorithm_radio, knn_method_radio, knn_metric_input,
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.knn import KNN
            clf = KNN(contamination=contamination, n_neighbors=neighbers, algorithm=algorithm,method=method, metric=metric)
            name = 'KNN ({},{},{},{})'.format(contamination,neighbers,algorithm,method,metric)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 10- ABOD Callback
    @app.callback(
        Output('abod_add_signal', 'data'),
        [Input("add_abod_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("abod_neighbers_input", "value"),
        State("abod_radio", "value"),]
    ) 
    def add_knn_clbk(n,contamination,neighbers,radio):#[knn_neighbers_input, knn_algorithm_radio, knn_method_radio, knn_metric_input,
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.abod import ABOD
            clf = ABOD(contamination=contamination, n_neighbors=neighbers, method=radio)
            name = 'ABOD ({},{},{})'.format(contamination,neighbers,radio)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 11- IForest Callback
    @app.callback(
        Output('iforest_add_signal', 'data'),
        [Input("add_iforest_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("iforest_estimators_input", "value"),
        State("iforest_switch", "value"),]
    ) 
    def add_iforest_clbk(n,contamination,estimators,bootstrap):#[knn_neighbers_input, knn_algorithm_radio, knn_method_radio, knn_metric_input,
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.iforest import IForest
            clf = IForest(contamination=contamination, n_estimators=estimators, bootstrap=(1 in bootstrap),random_state=random_state)
            name = 'IForest ({},{},{})'.format(contamination,estimators,(1 in bootstrap))
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 12- Feature Bagging Callback
    @app.callback(
        Output('fb_add_signal', 'data'),
        [Input("add_fb_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("fb_estimators_input", "value"),
        State("fb_radio", "value"),
        State("fb_switch", "value"),]
    ) 
    def add_fb_clbk(n,contamination,estimators,combination,bootstrap):#[knn_neighbers_input, knn_algorithm_radio, knn_method_radio, knn_metric_input,
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.feature_bagging import FeatureBagging
            clf = FeatureBagging(contamination=contamination, n_estimators=estimators, bootstrap_features=(1 in bootstrap),combination=combination , random_state=random_state)
            name = 'FB ({},{},{},{})'.format(contamination,estimators,combination,(1 in bootstrap))
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    # 13- SOD Callback
    @app.callback(
        Output('sod_add_signal', 'data'),
        [Input("add_sod_btn", "n_clicks")],
        [State("model_contamination", "value"),
        State("sod_neighbers_input", "value"),
        State("sod_refset_input", "value"),
        State("sod_alpha_input", "value"),]
    ) 
    def add_sod_clbk(n,contamination,neighbers,refset,alpha):#[knn_neighbers_input, knn_algorithm_radio, knn_method_radio, knn_metric_input,
        if (n is None):
            raise PreventUpdate
        else:
            from pyod.models.sod import SOD
            clf = SOD(contamination=contamination, n_neighbors=neighbers, ref_set=refset, alpha=alpha)
            name = 'SOD ({},{},{},{})'.format(contamination,neighbers,refset,alpha)
            DataStorage.model_list.append(emadModel(name,clf))
            # print(DataStorage.model_list) # Debug
            return '', {'added': True}

    @app.callback(
    Output('alert_list_saved', 'is_open'),
    [Input('save_list', 'n_clicks')]
    )
    def save_list(n):
        if (n is not None):
            from joblib import dump
            dump(DataStorage.model_list, 'list.joblib')
            return True
        else:
            return False

    @app.callback(
    Output('list_loaded', 'data'),
    [Input('load_list', 'n_clicks')]
    )
    def load_list(n):
        if (n is None):
            raise PreventUpdate
        else:
            from joblib import load
            DataStorage.model_list = load('list.joblib')
            return {'added': True}

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