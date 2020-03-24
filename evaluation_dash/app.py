# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from src.graphs import *

data_test = pd.read_csv("../data/data_test_example_0.csv")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


tabs_styles = {
    'height': '44px',
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    dcc.Tabs(style=tabs_styles, children=[
        ###########
        ## TAB 1 ##
        ###########
        dcc.Tab(label='Performance monitoring throughout the test', style=tab_style, selected_style=tab_selected_style,
                children=[
                            html.Div(children=[
                                                html.Div(children="Choose groups to compare",
                                                         style={'marginBottom': 25, 'marginTop': 50, 'fontSize': 25}),
                                                html.Div(children=dcc.Dropdown(id='my-id',
                                                                                options=groups_option_ab_test(data_test),
                                                                                multi=True,
                                                                                value=groups_option_ab_test(data_test)[0]["value"]
                                                                            )),
                                                dcc.Graph(id='my-div')
                                            ])
        ]),
        
        ###########
        ## TAB 2 ##
        ###########
        dcc.Tab(label='Accuracy of the results', style=tab_style, selected_style=tab_selected_style,
                children=[
                            html.Div(children="Choose the size of groups",
                                     style={'marginBottom': 25, 'marginTop': 50, 'fontSize': 25}
                                     ),
                            dcc.Slider(id='my-slider',
                                       min=1,
                                       max=1000,
                                       step=1,
                                       value=1,
                                       marks={1: '1 person',
                                              500: '500 persons',
                                              1000: '1000 persons'},
                                    ),
                            html.Div(id='proba_b_sup_c', style={'marginBottom': 5, 'marginTop': 5, 'fontSize': 25}
                                    ),
                            html.Div(children=dcc.Graph(id='slider-output-heatmap'),
                                     style={'marginLeft' : 500, 'marginBottom': 25, 'marginTop': 50, 'fontSize': 30, 'textAlign': 'center'}
                                     ),
                        ]),
                    ])
])
    

@app.callback(
    Output(component_id='my-div', component_property='figure'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return get_graph_follow_results(data_test, input_value)

@app.callback(
    Output(component_id='proba_b_sup_c', component_property='children'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output_error(input_value):
    probs = compute_error_probability_from_results(data_test["C"][:input_value], 
                                                   data_test["B"][:input_value])
    txt = "P[ perf B > perf C] = {}%".format(round(probs[0]*100,2))
    return txt

@app.callback(
    dash.dependencies.Output('slider-output-heatmap', 'figure'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output_heatmap(input_value):
    return heatmap_plotly_bayesian_test(data_test["C"][:input_value], 
                                        data_test["B"][:input_value], 
                                        name_a="C", 
                                        name_b="B")

if __name__ == '__main__':
    app.run_server(debug=True)