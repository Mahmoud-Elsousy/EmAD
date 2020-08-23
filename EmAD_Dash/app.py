import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import time
import numpy as np
import plotly.graph_objs as go

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

# from components.data_page import *
from components.deploy_page import *


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


if __name__ == "__main__":
    app.run_server(debug=True, port=4444, host='0.0.0.0')
