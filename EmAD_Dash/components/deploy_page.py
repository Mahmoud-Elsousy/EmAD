from components.test_page import *



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
                        dbc.Button("Download Model", id="download_model", size="lg",color="success",block=False, className="mx-2 mt-1"),
                        html.Hr(),
                        dbc.Button("Download Test.py", id="download_test", size="sm",outline=True, color="info",block=False, className="mx-4 mt-1"),
                        dbc.Button("Download Setup.sh", id="download_setup", size="sm",outline=True, color="info", className="mx-4 my-4"),

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
        return deploy_model_info_table()
