import plotly.express as px
import pandas as pd
# import numpy as np
# random_state = np.random.RandomState(3)

class DataStorage:
    loaded_data = {}
    xtr = pd.DataFrame()
    xte = pd.DataFrame()
    ytr = None
    yte = None
    model = {}

def update_scatter_matrix():
    fig = px.scatter_matrix(DataStorage.xtr,title="Scatter matrix of the Data", template="ggplot2", opacity=0.7)
    fig.update_traces(diagonal_visible=False)
    return fig

def update_line_plots():
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    num_of_features = DataStorage.xtr.shape[1]
    num_of_samples = DataStorage.xtr.shape[0]
    titles = DataStorage.xtr.columns
    fig = make_subplots(rows=num_of_features, cols=1)
    # fig.layout.plot_bgcolor = #fff
    
    for i in range(num_of_features):
        fig.add_trace(go.Scatter(x=list(range(0, num_of_samples)), y=DataStorage.xtr[titles[i]]),row=i+1, col=1)

    fig.update_layout(height=600, width=800, title_text="Feature line graphs",template="ggplot2")
    fig.layout.paper_bgcolor = 'rgba(220,220,220,0.3)'
    return fig


def process_loaded_data(df,generate_headers=False,shuffle=False, labels=None, nan=None,ratio=70):
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
        df.drop(df.columns[0],axis=1,inplace=True)
    elif labels=='last':
        y=df[df.columns[features-1]]
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
                      train_only=False, offset=offset, behaviour='new',random_state=None)

    DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = numpy_to_df(xtr, xte, ytr, yte)

    return {'loaded':True}

def generate_data_clusters(n_train, n_test, n_clusters, n_features, contamination,offset):
    from pyod.utils.data import generate_data_clusters

    xtr, xte, ytr, yte = generate_data_clusters(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination,
                      size='same',density='same', dist=(offset/40), random_state=None,return_in_clusters=False)

    DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = numpy_to_df(xtr, xte, ytr, yte)

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

    return {'loaded':True}


def load_link_to_df(link,header,shuffle, label, nan, ratio):
    import pandas as pd
    df = pd.read_csv(link)
    DataStorage.xtr, DataStorage.xte, DataStorage.ytr, DataStorage.yte = process_loaded_data(df
    ,header,shuffle, label, nan, ratio)


    return {'loaded':True}


'''Training models'''
def train_model_iforest(model_feature, contamination):
    if DataStorage.loaded_data is not None:
        from pyod.models.iforest import IForest
        DataStorage.model = IForest(contamination=contamination)
        DataStorage.model.fit(DataStorage.xtr)
        return {'trained':True}
    else:
        print("No data to train on")
        return {'trained':False}

def train_model_knn(model_feature, contamination):
    if DataStorage.loaded_data is not None:
        from pyod.models.knn import KNN
        DataStorage.model = KNN(contamination=contamination)
        DataStorage.model.fit(DataStorage.xtr)
        return {'trained':True}
    else:
        print("No data to train on")
        return {'trained':False}

def train_model_lof(n_neighbors, contamination):
    if DataStorage.loaded_data is not None:
        from pyod.models.lof import LOF
        DataStorage.model = LOF(n_neighbors=n_neighbors, contamination=contamination)
        DataStorage.model.fit(DataStorage.xtr)
        return {'trained':True}
    else:
        print("No data to train on")
        return {'trained':False}
