import plotly.express as px
import pandas as pd
# import numpy as np
# random_state = np.random.RandomState(3)

class DataStorage:
    loaded_data = {}
    model = {}

def update_scatter_matrix():
    fig = px.scatter_matrix(DataStorage.loaded_data['xtr'],title="Scatter matrix of the Data")
    fig.update_traces(diagonal_visible=False)
    return fig

def update_line_plots():
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    num_of_features = DataStorage.loaded_data['xtr'].shape[1]
    num_of_samples = DataStorage.loaded_data['xtr'].shape[0]
    titles = DataStorage.loaded_data['xtr'].columns
    fig = make_subplots(rows=num_of_features, cols=1)
    for i in range(num_of_features):
        fig.add_trace(go.Scatter(x=list(range(0, num_of_samples)), y=DataStorage.loaded_data['xtr'][titles[i]]),row=i+1, col=1)

    fig.update_layout(height=600, width=800, title_text="Feature line graphs")
    return fig


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

    xtr, xte, ytr, yte = numpy_to_df(xtr, xte, ytr, yte)

    data = {
    'xtr':xtr,
    'xte':xte,
    'ytr':ytr,
    'yte':yte,}
    DataStorage.loaded_data = data
    return {'loaded':True}

def generate_data_clusters(n_train, n_test, n_clusters, n_features, contamination,offset):
    from pyod.utils.data import generate_data_clusters

    xtr, xte, ytr, yte = generate_data_clusters(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination,
                      size='same',density='same', dist=(offset/40), random_state=None,return_in_clusters=False)

    data = {
    'xtr':xtr,
    'xte':xte,
    'ytr':ytr,
    'yte':yte,}
    DataStorage.loaded_data = data
    return {'loaded':True}



def load_file_to_df(contents, filename, training_data_ratio):
    import pandas as pd
    import base64
    import io

    for content, name in zip(contents, filename):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                split_row = int((training_data_ratio/100)*df.shape[0])
                xtr = df[:split_row]
                xte = df[split_row:]
                data = {
                'xtr':xtr,
                'xte':xte,
                }
                DataStorage.loaded_data = data

            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
                split_row = int((training_data_ratio/100)*df.shape[0])
                xtr = df[:split_row]
                xte = df[split_row:]
                data = {
                'xtr':xtr,
                'xte':xte,
                }
                DataStorage.loaded_data = data
        except Exception as e:
            print(e)

    return {'loaded':True}


def load_link_to_df(link,training_data_ratio):
    import pandas as pd
    df = pd.read_csv(link)
    split_row = int((training_data_ratio/100)*df.shape[0])
    xtr = df[:split_row]
    xte = df[split_row:]
    data = {'xtr':xtr,'xte':xte,}
    DataStorage.loaded_data = data


    return {'loaded':True}


'''Training models'''
def train_model_iforest(model_feature, contamination):
    if DataStorage.loaded_data is not None:
        from pyod.models.iforest import IForest
        DataStorage.model = IForest(contamination=contamination)
        DataStorage.model.fit(DataStorage.loaded_data['xtr'])
        return {'trained':True}
    else:
        print("No data to train on")
        return {'trained':False}

def train_model_knn(model_feature, contamination):
    if DataStorage.loaded_data is not None:
        from pyod.models.knn import KNN
        DataStorage.model = KNN(contamination=contamination)
        DataStorage.model.fit(DataStorage.loaded_data['xtr'])
        return {'trained':True}
    else:
        print("No data to train on")
        return {'trained':False}

def train_model_lof(n_neighbors, contamination):
    if DataStorage.loaded_data is not None:
        from pyod.models.lof import LOF
        DataStorage.model = LOF(n_neighbors=n_neighbors, contamination=contamination)
        DataStorage.model.fit(DataStorage.loaded_data['xtr'])
        return {'trained':True}
    else:
        print("No data to train on")
        return {'trained':False}
