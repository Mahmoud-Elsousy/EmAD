from components.test_page import *
import os


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
        dump(DataStorage.deploy_model, 'model.joblib')
        dump(DataStorage.xte.values, 'xte.joblib')
        if DataStorage.yte is not None:
            dump(DataStorage.yte.values, 'yte.joblib')
        return deploy_model_info_table()
