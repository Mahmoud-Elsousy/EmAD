import plotly.express as px
import pandas as pd
from io import StringIO
import dash_html_components as html
import dash_bootstrap_components as dbc
from pyod.models.base import BaseDetector
import numpy as np
random_state = np.random.RandomState(3)
from time import time


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
    import sys

    for mod in DataStorage.model_list:
        t=time() 
        mod.clf.fit(DataStorage.xtr)
        t = time() - t
        p = pickle.dumps(mod.clf)
        mod.size = sys.getsizeof(p)/1000
        mod.isTrained = 'Yes'
        mod.training_time = t


def generate_test_table():
    table_header = [html.Thead(html.Tr([html.Th("#"),
     html.Th("Model"),
     html.Th("Trained?"),
     html.Th("Size(KB)"),
     html.Th("Training Time(s)"),
     html.Th("Inference Time(ms)"),
     html.Th("AUC Score"),
     html.Th("Precision @ n "),
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
         html.Td('%.4f'%(mod.inference_time)),
         html.Td('%.3f'%(mod.auc)),
         html.Td('%.3f'%(mod.pan)),
         html.Td(mod.tn),
         html.Td(mod.fp),
         html.Td(mod.fn),
         html.Td(mod.tp),
         ]))

    table_body = [html.Tbody(rows)]
    return dbc.Table(table_header + table_body, bordered=True, hover=True,responsive=True)


def test_models():
    from pyod.utils.utility import precision_n_scores
    from sklearn.metrics import roc_auc_score, confusion_matrix
    xte = DataStorage.xte.to_numpy()
    for mod in DataStorage.model_list:
        t=time()
        scores = mod.clf.decision_function(xte) 
        mod.inference_time = ((time() - t)*1000)/np.shape(xte)[0]
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
    row4 = html.Tr([html.Td("Inference Time(ms)"), html.Td(('%.4f'%dm.inference_time))])
    row5 = html.Tr([html.Td("AUC Score"), html.Td('%%%.1f'%(dm.auc*100))])
    row6 = html.Tr([html.Td("P@n Score"), html.Td('%%%.1f'%(dm.pan*100))])

    table_body = [html.Tbody([row1, row2, row3, row4, row5, row6])]
    table = dbc.Table(table_body, bordered=False,)
    cont = dbc.Container([model_name,table])

    return cont