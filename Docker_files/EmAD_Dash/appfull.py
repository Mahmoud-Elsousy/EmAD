import os
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
import plotly.express as px
app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

import pandas as pd
import numpy as np

from io import StringIO

random_state = np.random.RandomState(3)
from time import time

dn = os.path.abspath(os.getcwd())
run_path = '/emad/EmAD_Dash/'

# run_path = ''
# print('Path: ', dn)
class emadModel:
    def __init__(self, name, clf):
        self.name = name
        self.clf = clf
        self.isTrained = 'No'
        self.n_features = 0
        self.size=0
        self.training_time = 0
        self.inference_time = 0
        self.auc = 0
        self.pan = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.b1 = 0
        self.b10 = 0
        self.b100 = 0
        self.b1000 = 0



class DataStorage:
    loaded_data = {}
    xtr = pd.DataFrame()
    xte = pd.DataFrame()
    ytr = None
    yte = None
    source_name='' # Data Source Name
    model_list = []
    deploy_model = emadModel('temp', None)
         

def update_scatter_matrix(dfx,dfy):
    if dfy is None: 
        # The case when there are no labels      
        fig = px.scatter_matrix(dfx, template="plotly_white", opacity=0.7)
        fig.update_traces(diagonal_visible=False)
    else:

        label=dfy.Y.replace({1: 'Anomaly', 0: 'Normal'})
        fig = px.scatter_matrix(dfx,dimensions= dfx.columns,color=label,
        template="plotly_white", opacity=0.7)
        fig.update_traces(diagonal_visible=False)

    return fig

def update_line_plots(dfx=DataStorage.xtr):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    num_of_features = dfx.shape[1]
    num_of_samples = dfx.shape[0]
    titles = dfx.columns
    fig = make_subplots(rows=num_of_features, cols=1)
    # fig.layout.plot_bgcolor = #fff
    
    for i in range(num_of_features):
        fig.add_trace(go.Scatter(x=list(range(0, num_of_samples)), y=dfx[titles[i]], 
        name=titles[i]),row=i+1, col=1)

    fig.update_layout(height=600, width=800,template="plotly_white")
    # fig.layout.paper_bgcolor = 'rgba(220,220,220,0.3)'
    return fig


def process_loaded_data(df,generate_headers=False,shuffle=False, labels=None, nan=None,ratio=70, random_state=random_state):
    from sklearn.model_selection import train_test_split
    features = df.shape[1]

    # Generate headers if needed
    if generate_headers:
        headers = []
        for i in range(features):
            headers.append('F'+str(i))
        df.columns = headers

    # If a label column is selected separate the labels from data
    y= None
    if labels=='first':
        y=df[df.columns[0]]
        y.columns=['Y']
        df.drop(df.columns[0],axis=1,inplace=True)
    elif labels=='last':
        y=df[df.columns[features-1]]
        y.columns=['Y']
        df.drop(df.columns[features-1],axis=1,inplace=True)

    # Process missing values
    if nan == 'ffill':
        df.fillna(method='ffill', inplace=True)
    elif nan == 'zero':
        df.fillna(0, inplace=True)
    elif nan == 'delete':
        df.dropna(how = 'all', inplace=True)

    if y is not None:
        xtr, xte, ytr, yte = train_test_split(df, y, test_size=(100-ratio)/100,shuffle=shuffle) 
        # The following step had to be calculated after the split to avoid data leakage betwean training and testing data.
        if nan == 'average':
            xtr.fillna(xtr.mean(), inplace=True)
            xte.fillna(xte.mean(), inplace=True)
        return xtr, xte, ytr, yte
    else:
        xtr, xte = train_test_split(df, test_size=(100-ratio)/100,shuffle=shuffle)
        return  xtr, xte, None, None



def numpy_to_df(xtr, xte, ytr, yte):

    columns =[]
    for i in range(xtr.shape[1]):
        columns.append("F"+str(i))

    xtr = pd.DataFrame(xtr,columns=columns)
    xte = pd.DataFrame(xte,columns=columns)
    ytr = pd.DataFrame(ytr,columns=['Y'],dtype='int8')
    yte = pd.DataFrame(yte,columns=['Y'],dtype='int8')
    return xtr, xte, ytr, yte

def generate_data_simple(n_train, n_test, n_features, contamination,offset):
    from pyod.utils.data import generate_data

    xtr, xte, ytr, yte = generate_data(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination,
                      train_only=False, offset=offset, behaviour='new',random_state=random_state)
    
    DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = numpy_to_df(xtr, xte, ytr, yte)
    DataStorage.source_name = 'Generated Data - Simple'
    return {'loaded':True}

def generate_data_clusters(n_train, n_test, n_clusters, n_features, contamination,offset):
    from pyod.utils.data import generate_data_clusters

    xtr, xte, ytr, yte = generate_data_clusters(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination,
                      n_clusters=n_clusters,size='same',density='same', dist=(offset/40), random_state=random_state,return_in_clusters=False)

    DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = numpy_to_df(xtr, xte, ytr, yte)
    DataStorage.source_name = 'Generated Data - Clusters'
    return {'loaded':True}



def load_file_to_df(contents, filename,header,shuffle, label, nan, ratio):
    import pandas as pd
    import base64
    import io

    # for content, name in zip(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = process_loaded_data(df
            ,header,shuffle, label, nan, ratio)
        elif 'xls' in filename:
            # Assume that user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = process_loaded_data(df
            ,header,shuffle, label, nan, ratio)

    except Exception as e:
        print(e)
        DataStorage.source_name=''
        return {'loaded':False}
    
    DataStorage.source_name=filename
    return {'loaded':True}


def load_link_to_df(link,header,shuffle, label, nan, ratio):
    import pandas as pd
    try:
        df = pd.read_csv(link)
        DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = process_loaded_data(df
        ,header,shuffle, label, nan, ratio)
    except Exception as e:
        print(e)
        DataStorage.source_name=''
        return {'loaded':False}       

    DataStorage.source_name=link.split('/')[-1]
    return {'loaded':True}


def get_data_info():
    tr_info = StringIO()
    te_info = StringIO()
    DataStorage.xtr.info(buf=tr_info)
    DataStorage.xte.info(buf=te_info)

    training = []
    for i in tr_info.getvalue().split('\n')[1:]:
        training.append(html.Br())
        training.append(i)

    testing = []
    for i in te_info.getvalue().split('\n')[1:]:
        testing.append(html.Br())
        testing.append(i)

    info = dbc.Container([html.H3(DataStorage.source_name),html.Br(),
    dbc.Row([
        dbc.Col([
            html.H4('Training Data:',className = 'text-secondary'),
            html.Code(training,className = 'text-primary'),
        ]),
        dbc.Col([
            html.H4('Testing Data:',className = 'text-secondary'),
            html.Code(testing,className = 'text-primary'),
        ])
    ]),
    ], className='text-muted')
    return info

def generate_model_table():
    table_header = [html.Thead(html.Tr([html.Th("#"),
     html.Th("Model"),
     html.Th("Trained?"),
     html.Th("Size(KB)"),
     html.Th("Training Time(s)")]))
    ]
    rows = []
    i=0
    for mod in DataStorage.model_list:
        i+=1
        rows.append(html.Tr([html.Td(i),
         html.Td(mod.name),
         html.Td(mod.isTrained),
         html.Td('%.2f'%(mod.size)),
         html.Td('%.2f'%(mod.training_time)),
         ]))

    table_body = [html.Tbody(rows)]
    return dbc.Table(table_header + table_body, striped=True, bordered=True, hover=True,responsive=True)

def train_models():
    import pickle
    from joblib import dump
    import sys

    i=0
    for mod in DataStorage.model_list:
        t=time() 
        mod.clf.fit(DataStorage.xtr)
        t = time() - t
        p = pickle.dumps(mod.clf)
        mod.size = sys.getsizeof(p)/1000
        i+=1
        # dump(mod.clf, 'mod%d.clf'%(i))
        mod.isTrained = 'Yes'
        mod.training_time = t


def generate_test_table():
    table_header = [html.Thead(html.Tr([html.Th("#"),
     html.Th("Model"),
     html.Th("Trained?"),
     html.Th("Size(KB)"),
     html.Th("Training Time(s)"),
     html.Th("AUC Score"),
     html.Th("Precision @ n "),
     html.Th("B1"),
     html.Th("B10"),
     html.Th("B100"),
     html.Th("B1000"),
     html.Th("TN"),
     html.Th("FP"),
     html.Th("FN"),
     html.Th("TP"),
     ]))
    ]
    rows = []
    i=0
    for mod in DataStorage.model_list:
        i+=1
        rows.append(html.Tr([html.Td(i),
         html.Td(mod.name),
         html.Td(mod.isTrained),
         html.Td('%.2f'%(mod.size)),
         html.Td('%.2f'%(mod.training_time)),
         html.Td('%.3f'%(mod.auc)),
         html.Td('%.3f'%(mod.pan)),
         html.Td('%.3f'%(mod.b1)),
         html.Td('%.3f'%(mod.b10)),
         html.Td('%.3f'%(mod.b100)),
         html.Td('%.5f'%(mod.b1000)),
         html.Td(mod.tn),
         html.Td(mod.fp),
         html.Td(mod.fn),
         html.Td(mod.tp),
         ]))

    table_body = [html.Tbody(rows)]
    return dbc.Table(table_header + table_body, bordered=True, hover=True,responsive=True)


def batch_inference_time(model, xte, batch_size=1):
    average_time = 0
    n_batches = int(xte.shape[0]/batch_size)
    for i in range(n_batches):
        batch = xte[(i*batch_size):((i+1)*batch_size),:]
        t=time()
        model.predict(batch)
        inference_time = ((time() - t)*1000)/batch_size
        average_time += inference_time
    
    return average_time/n_batches


def test_models():
    from pyod.utils.utility import precision_n_scores
    from sklearn.metrics import roc_auc_score, confusion_matrix
    xte = DataStorage.xte.values
    for mod in DataStorage.model_list:
        if 'COF' not in mod.name:
            mod.b1 = batch_inference_time(mod.clf,xte,1)
            mod.b10 = batch_inference_time(mod.clf,xte,10)
        mod.b100 = batch_inference_time(mod.clf,xte,100)
        mod.b1000 = batch_inference_time(mod.clf,xte,1000)
        print('1:%f,10:%f,100:%f,1000:%f'%(mod.b1,mod.b10,mod.b100,mod.b1000))
        scores = mod.clf.decision_function(xte) 
        print(scores)
        mod.auc = roc_auc_score(DataStorage.yte, scores)
        mod.pan = precision_n_scores(DataStorage.yte, scores)
        y_pre = mod.clf.predict(xte)
        mod.tn, mod.fp, mod.fn, mod.tp = confusion_matrix(DataStorage.yte,y_pre).ravel()


def deploy_model_info_table():
    dm = DataStorage.deploy_model

    model_name = html.H4(dm.name, className='text-secondary text-center mb-3')

    row1 = html.Tr([html.Td("Features #"), html.Td(dm.n_features)])
    row2 = html.Tr([html.Td("Size (Before Serialization)"), html.Td('%.3f'%(dm.size))])
    row3 = html.Tr([html.Td("Training Time(s)"), html.Td('%.3f'%(dm.training_time))])
    row4 = html.Tr([html.Td("Inference Time/Sample(B100)(ms)"), html.Td(('%.4f'%dm.b100))])
    row5 = html.Tr([html.Td("AUC Score"), html.Td('%%%.1f'%(dm.auc*100))])
    row6 = html.Tr([html.Td("P@n Score"), html.Td('%%%.1f'%(dm.pan*100))])

    table_body = [html.Tbody([row1, row2, row3, row4, row5, row6])]
    table = dbc.Table(table_body, bordered=False,)
    cont = dbc.Container([model_name,table])

    return cont

''' Defining the Data generator interface '''
dcc.Store(id="refresh_figures")
''' A- Define the Generate data card '''
label_width = 6
input_width = 5
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
    dbc.Col(dbc.Input(type="number", id="n_test", value=1000, step=100,),width=input_width,className="m-1"),],
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
    dbc.Col(dbc.Input(type="number", id="offset", value=1, step=1,),width=input_width,className="m-1"),],
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
            dbc.Tab(label="Table", tab_id="table"),
            dbc.Tab(label="Scatter", tab_id="scatter"),
            dbc.Tab(label="Line Plots", tab_id="line"),
            dbc.Tab(label="Info", tab_id="info"),
        ],
        id="data-graph-tabs",
        active_tab="table",
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
            duration=2000,
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
    children=html.Div(id="upload_box_text",children=['Drag and Drop or ',html.A('Select Files')]),
    style=UPLOAD_STYLE,
    multiple=False,
    className="btn btn-outline-primary mx-auto"
)

load_link_form = dbc.FormGroup(
    [dbc.Label("Link:", html_for="load_link", width=label_width-2,
    className="pl-4 pt-3", ),
    dbc.Col(dbc.Input(type="url", id="load_link",placeholder="https://github..."),width=input_width+2,
    className="pt-3"),],
    row=True,)

nan_radio = dbc.FormGroup(
    [
        dbc.Label("Missing values:"),
        dbc.RadioItems(
            options=[
                {"label": "Copy last valid", "value": 'ffill'},
                {"label": "Fill average", "value": 'average'},
                {"label": "Fill 0", "value": 'zero'},
                {"label": "Delete Sample", "value": 'delete'},
            ],
            value='ffill',
            id="nan_radio",
            className="px-3",
            # inline=True,
        ),
        # dbc.Row(dbc.Col(dbc.Input(id="nan_fill_value", value=0, type="number"),width=5,className="ml-auto mr-3")),
    ]
)

label_radio = dbc.FormGroup(
    [
        dbc.Label("Label Column:"),
        dbc.RadioItems(
            options=[
                {"label": "No Labels", "value": 'none'},
                {"label": "First", "value": 'first'},
                {"label": "Last", "value": 'last'},
            ],
            value='none',
            id="label_radio",
            className="px-3",
            inline=True,
        ),
    ]
)

load_switches = dbc.Checklist(
options=[{"label": "Generate Header", "value": 1},
{"label": "Shuffel Data", "value": 2},
],
value=[1],
id="load-switches",
switch=True,)

split_form = html.Div(dbc.FormGroup(
    [dbc.Label("Split % for Trainig:", html_for="split_data", width=label_width-1,className="pl-4"),
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
load_link_form,
html.Hr(),
load_switches,
html.Hr(),
label_radio,
nan_radio,
split_form,
dbc.Button("Load File", id="load",  color="success",block=True, className="mx-2 mt-1")
],body=True,
className="mt-3")


''' Defining the Data generator interface '''

data_page_container = html.Div(
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
                    active_tab="generate",
                    card=True
                )
        ), dbc.CardBody(
        html.Div(id="data-tab-content", children="Test")
        )
        ]),      
    ]
)

def data_tabs_callbacks(app):
    '''Data Generation card callbacks'''
    @app.callback(
    [Output('n_clusters', 'disabled'),
    Output('n_clusters', 'value')],
    [Input('select-generator', 'value')]
    )
    def update_generate_interface(value):
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
    State('label_radio', 'value'),
    State('nan_radio', 'value'),
    State('split_data', 'value')]
    )
    def load_data(n,contents, file_name, file_link,switch,label_radio, nan_radio, training_data_ratio):
        if n is None:
            raise PreventUpdate

        elif file_name is not None:
            data = load_file_to_df(contents, file_name,1 in switch,2 in switch, label_radio, nan_radio, training_data_ratio)
            return data, True

        elif (file_link is not None) and (len(file_link.split('.')) > 1):
            data = load_link_to_df(file_link,1 in switch,2 in switch, label_radio, nan_radio, training_data_ratio)
            return data, True

    '''Data Load card callbacks'''

    @app.callback(
        Output("data-graph-tab-content", "children"),
        [Input("data-graph-tabs", "active_tab"),
        Input("loaded_data_store", "data"),
        Input("generated_data_store", "data")],
    )
    def render_tab_graphs(active_tab, loaded,generated):

        loaded = loaded or {'loaded':False}
        generated = generated or {'loaded':False}

        if (active_tab is not None):
            if loaded['loaded'] is True:
                df = DataStorage.xtr
                dfy = DataStorage.ytr
            elif generated['loaded'] is True:
                df = DataStorage.xtr
                dfy = DataStorage.ytr
            else:
                return ''#html.Center(html.H3("No Data to represent!"), className='text-muted')

            # Return the contents based on the selected tab
            if active_tab == "scatter":
                return dcc.Graph(figure=update_scatter_matrix(df,dfy))
            
            elif active_tab == "line":
                return dcc.Graph(figure=update_line_plots(df))
            
            elif active_tab == "info":
                info = get_data_info()
                return info
            
            elif active_tab == "table":
                return dt.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                page_size=15,
                style_cell={'textAlign': 'left'},
                # fixed_rows={'headers': True},
                )

        return "No tab selected"

    '''Data main tabs control'''
    @app.callback(
        Output("data-tab-content", "children"),
        [Input("data-tabs", "active_tab")]
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
                    dbc.Col(dbc.Button(href="/page-2",disabled=True, id='btn_to_training', className="mx-auto p-2 btn-success", block=True,),
                     width=6, className="mx-auto my-3"),
                ])
                ], className='px-0 mx-0')
                return content
            
            
            if active_tab == "generate":
                return one_third_two_thirds(generate_card,data_graphs_div)

            elif active_tab == "load":               
                return one_third_two_thirds(load_card,data_graphs_div)


        return "No tab selected"

    @app.callback(
        [Output("upload_box_text", "children"),
        Output("load", "disabled"), 
        Output("load", "children"),],
        [Input("upload-data", "filename"),
        Input("load_link", "value")]
    ) 
    def display_file_name(filename, link_value):
        load_text_content = ['Drag and Drop or ',html.A('Select Files')]
        load_btn_disabled = True
        btn_text = "Select Data to Load"

        if filename is not None:
            btn_text = "Press to Load: " + str(filename)
            load_text_content = html.P(filename, className="p-auto")
            load_btn_disabled = False

        # Just a basic check that the of the box is not empty and has at least a .
        elif (link_value is not None) and (len(link_value.split('.')) > 1):
            btn_text = "Press to Load from Link"
            load_btn_disabled = False

        return load_text_content, load_btn_disabled, btn_text 
 

    @app.callback(
        [Output("btn_to_training", "children"),
        Output("btn_to_training", "disabled")],
        [Input("generated_data_store", "data"),
        Input("loaded_data_store", "data")]
    ) 
    def activate_to_train_btn(data1, data2):
        if (data1 is not None) or (data2 is not None):
            return "2- Model Training" , False
        
        return "Please load data to train on!", True

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

clear_list_btn = dbc.Button("Clear List", id="clear_list_btn",outline=True, size="sm", color="warning",block=True, className="my-2 px-4")

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
clear_list_btn,
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
        Input('list_cleared', 'data'),
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
    Output('list_cleared', 'data'),
    [Input('clear_list_btn', 'n_clicks')]
    )
    def clear_list(n):
        if (n is None):
            raise PreventUpdate
        else:
            DataStorage.model_list = []
            return {'cleared': True}


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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

''' Defining the Model training and testing interface '''

test_page_container = html.Div(
    [
        html.H1("3- Model Testing"),
        dbc.Card([
        dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Model Testing", tab_id="model_testing_tab",),
                        dbc.Tab(label="Testing Data", tab_id="testing_data_tab",),
                    ],
                    id="test-tabs",
                    active_tab="model_testing_tab",
                    card=True
                )
        ), dbc.CardBody(
        html.Div(id="test-tab-content", children="Callback Failed", className='px-0 mx-0')
        )
        ])

    ]
)

# Define test data graphs
test_graphs_tabs = dbc.Tabs(
        [      
            dbc.Tab(label="Table", tab_id="test_table"),
            dbc.Tab(label="Scatter", tab_id="test_scatter"),
            dbc.Tab(label="Line Plots", tab_id="test_line"),
            dbc.Tab(label="Info", tab_id="test_info"),
        ],
        id="test-graph-tabs",active_tab="test_table",)


# Define deploy model

def test_tabs_callbacks(app):
    '''Data main tabs control'''
    @app.callback(
        Output("test-tab-content", "children"),
        [Input("test-tabs", "active_tab")]
    )
    def render_test_tab_content(active_tab):
        if active_tab is not None:
            def one_block(inner_content):
                content = dbc.Container([
                dbc.Row(inner_content),
                dbc.Row([
                    dbc.Col(dbc.Button("2- Model Training",href="/page-2", id='btn_to_train', className="mx-auto p-2 btn-secondary", block=True,),
                     width=4, className="ml-auto mt-4"),
                    dbc.Col(dbc.Button("Test and Choose a model to Deploy",href="/page-4",disabled=True, id='btn_to_deployment', className="mx-auto p-2 btn-success", block=True,),
                     width=4, className="mr-auto mt-4"),
                ])
                ], className='px-0 mx-0')
                return content

            if active_tab == "model_testing_tab":
                test_models_card=dbc.Card([
                html.H5("Trained Models", className="text-primary mx-auto"),
                html.Hr(),
                dbc.Row(id='test_models_table'),
                dbc.Row(dbc.Col(dbc.Spinner(html.Div(id='test_loading', className='my-3')), width=12)),
                dbc.Row(dbc.Button("Start Testing", id='btn_to_test', className="mx-auto p-2 btn-success", block=True,)),
                ],body=True,
                className="my-3")
                return one_block(test_models_card)

            elif active_tab == "testing_data_tab":
                test_data_card=dbc.Card([
                # html.H5("Testing Data", className="text-primary mx-auto"),
                # html.Hr(),
                test_graphs_tabs,
                html.Div(id="test-graph-tab-content", className="p-0"),
                ],body=True,
                className="my-3")
                return one_block(test_data_card)

        return "No tab selected"


    @app.callback(
        Output("test-graph-tab-content", "children"),
        [Input("test-graph-tabs", "active_tab")]
    )
    def render_test_graphs(active_tab):
        if (active_tab is not None):
            df = DataStorage.xte
            dfy = DataStorage.yte

            if active_tab == "test_scatter":
                return dcc.Graph(figure=update_scatter_matrix(df,dfy))
            
            elif active_tab == "test_line":
                return dcc.Graph(figure=update_line_plots(df))
            
            elif active_tab == "test_info":
                info = get_data_info()
                return info
            
            elif active_tab == "test_table":
                return dt.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                page_size=15,
                style_cell={'textAlign': 'left'},
                # fixed_rows={'headers': True},
                )

        return "No tab selected"
   

    @app.callback(
    [Output('test_loading', 'children'),
    Output('test_signal', 'data'),
    Output('btn_to_deployment', 'disabled'),
    Output('btn_to_deployment', 'children')],
    [Input('btn_to_test', 'n_clicks')]
    )
    def test_added_models(n):
        if (n is None):
            return '', {'added': False}, True, 'Test and select a model to deploy'
        else:
            test_models()
            tested_models = []
            for i in range(len(DataStorage.model_list)):
                tested_models.append({'label': DataStorage.model_list[i].name, 'value': i})
            deploy_dropdown = dcc.Dropdown(id='deploy_model',options=tested_models,value=0),
            return deploy_dropdown, {'added': True}, False, '4- Deploy Selected Model'

    @app.callback(
        Output("test_models_table", "children"),
        [Input("test_signal", "data")]
    )
    def render_test_table(data):
        return generate_test_table()

    @app.callback(
        Output('deploy_signal', 'data'),
        [Input("deploy_model", "value")]
    )
    def save_deploy_model(val):
        DataStorage.deploy_model = DataStorage.model_list[val]
        DataStorage.deploy_model.n_features = DataStorage.xtr.shape[1]
        return {'selected':True}

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

''' Defining the Model training and testing interface '''

deploy_page_container = html.Div(
    [
        html.H1("4- Deployment"),
        dbc.Card([
            dbc.Row(
                dbc.Col(
                    dbc.Card([
                        html.H5("Deploy Model", className="text-primary mx-auto mt-4"),
                        html.Hr(),
                        # deploy_model_info_table(),
                        html.Div(id='deploy_model_info'),
                        html.A(dbc.Button("Download Model", id="download_model", size="lg",color="success",block=True, className="mt-1 px-2",),
                        href="/model.joblib", className="px-2 mx-0"),
                            
                        html.Hr(),

                        html.A(dbc.Button("Download example.py", id="download_example", size="sm",outline=True,color="info",block=True, className="my-2 px-1",),
                        href='/example.py', className="px-4 mx-0"),

                        html.A(dbc.Button("test_model.py", id="download_test_model", size="sm",outline=True,color="info",block=True, className="my-2 px-1",),
                                href='/test_model.py', className="px-4 mx-0"),
                        
                        dbc.Row([
                            dbc.Col(
                                html.A(dbc.Button("Xtest", id="download_xte", size="sm",outline=True,color="info",block=True, className="my-1 pl-1",),
                                href='/xte.joblib', className="px-0 mx-0"),
                                className="pl-4"),
                            dbc.Col(
                                html.A(dbc.Button("Ytest", id="download_yte", size="sm",outline=True,color="info",block=True, className="my-1 pr-1",),
                                href='/yte.joblib', className="px-0 mx-0"),
                                className="pr-4"),
                        ],className="px-0 mx-0"),

                        html.A(dbc.Button("Download setup.sh", id="download_setup", size="sm",outline=True,color="warning",block=True, className="mt-1 mb-4 px-4",),
                        href='/setup.sh', className="px-4 mx-0"),

                    ], color="success", outline=True)
                ,width={"size": 4, "offset": 4})
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Button("3- Model Testing",href="/page-3", id='btn_back_test', className="mx-auto p-2 btn-secondary my-3", block=True,)
                ,width={"size": 4, "offset": 4})
            )

        ], body=True)

    ]
)




def deploy_callbacks(app):
   
    @app.callback(
        Output('deploy_model_info', 'children'),
        [Input("deploy_signal", "data")]
    )
    def save_deploy_model(val):
        from joblib import dump
        dump(DataStorage.deploy_model, os.path.join(dn,"model.joblib"))
        dump(DataStorage.xte.values, os.path.join(dn,"xte.joblib"))
        if DataStorage.yte is not None:
            dump(DataStorage.yte.values,os.path.join(dn,"yte.joblib"))
        return deploy_model_info_table()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

data_tabs_callbacks(app)
model_tabs_callbacks(app)
test_tabs_callbacks(app)
deploy_callbacks(app)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",

}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("EmAD", className="display-4"),
        html.Hr(),
        html.P(
            "Anomaly detection framework for ARM based Embedded Systems", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("1. Data Preparation", href="/page-1", id="page-1-link"),
                dbc.NavLink("2. Model Training", href="/page-2", id="page-2-link"),
                dbc.NavLink("3. Model Testing", href="/page-3", id="page-3-link"),
                dbc.NavLink("4. Deployment", href="/page-4", id="page-4-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content,
dcc.Store(id="generated_data_store"),
dcc.Store(id="added_model_store"),
dcc.Store(id="loaded_model_store"),
dcc.Store(id='pca_add_signal'),
dcc.Store(id='mcd_add_signal'),
dcc.Store(id='ocsvm_add_signal'),
dcc.Store(id='lmdd_add_signal'),
dcc.Store(id='lof_add_signal'),
dcc.Store(id='cof_add_signal'),
dcc.Store(id='cblof_add_signal'),
dcc.Store(id='hbos_add_signal'),
dcc.Store(id='knn_add_signal'),
dcc.Store(id='abod_add_signal'),
dcc.Store(id='iforest_add_signal'),
dcc.Store(id='fb_add_signal'),
dcc.Store(id='sod_add_signal'),
dcc.Store(id='list_loaded'),
dcc.Store(id='list_cleared'),
dcc.Store(id='train_signal'),
dcc.Store(id='test_signal'),
dcc.Store(id='deploy_signal'),
dcc.Store(id="loaded_data_store")])


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 5)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False
    return [pathname == f"/page-{i}" for i in range(1, 5)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return data_page_container
    elif pathname == "/page-2":
        return model_page_container
    elif pathname == "/page-3":
        return test_page_container
    elif pathname == "/page-4":
        return deploy_page_container
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

from flask import send_file
@app.server.route("/example.py") 
def send_example():
    return send_file(run_path + 'example.py',as_attachment=True)

@app.server.route("/setup.sh") 
def send_setup():
    return send_file(run_path + 'setup.sh',as_attachment=True)

@app.server.route("/test_model.py") 
def send_test_model():
    return send_file(run_path + 'test_model.py',as_attachment=True)

@app.server.route("/xte.joblib") 
def send_xte():
    return send_file(os.path.join(dn,"xte.joblib"),as_attachment=True)

@app.server.route("/yte.joblib") 
def send_yte():
    return send_file(os.path.join(dn,"yte.joblib"),as_attachment=True)

@app.server.route("/model.joblib") 
def send_model():
    return send_file(os.path.join(dn,"model.joblib"),as_attachment=True)

if __name__ == "__main__":
    app.run_server(port=4444, host='0.0.0.0')
    # app.run_server(debug=True, port=4444, host='0.0.0.0')
