

# import numpy as np
# random_state = np.random.RandomState(3)

def df_to_dict(df):
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),)

def numpy_to_dataframes(xtr, xte, ytr, yte):
    import pandas as pd
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
    data = {
    'loaded':True,
    'xtr':xtr,
    'xte':xte,
    'ytr':ytr,
    'yte':yte,}
    return data

def generate_data_clusters(n_train, n_test, n_clusters, n_features, contamination,offset):
    from pyod.utils.data import generate_data_clusters

    xtr, xte, ytr, yte = generate_data_clusters(n_train=n_train, n_test=n_test, n_features=n_features, contamination=contamination,
                      size='same',density='same', dist=(offset/40), random_state=None,return_in_clusters=False)
    data = {
    'loaded':True,
    'name':"Generated Data",
    'xtr':xtr,
    'xte':xte,
    'ytr':ytr,
    'yte':yte,}
    return data


def load_file_to_dict(contents, filename, training_data_ratio):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return "file type not supported"

    split_row = int((training_data_ratio/100)*df.shape[0])
    xtr = df[:split_row]
    xte = df[split_row:]
    data = {
    'loaded':1,
    'name':filename,
    'xtr':xtr,
    'xte':xte,
    }
    return data
