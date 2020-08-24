from components.model_page import *



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
