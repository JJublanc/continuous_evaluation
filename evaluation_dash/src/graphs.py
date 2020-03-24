import numpy as np
import plotly.graph_objects as go
from src.bayesian_test import *

def cum_mean(vect):
    """
    Compute the cumulative mean for a given vectore of numbers
    Args:
        vect:list or numpy array of int or float
    Return:
        vect or list
    """
    cum_sum = np.cumsum(vect)
    cum_mean = [cum_sum[i] / (i + 1) for i in range(len(vect))]
    return cum_mean


def get_graph_follow_results(data_test, list_var):
    """
    create the graph that allow one to follow the cumulative means for selected groups of the test
    Arg:
        list_var: list of the groups' name that one want to plot
    Return:
        a figure plotly
    """
    fig = go.Figure()
    
    for group in list_var:
        print(group)
        size = len(data_test[group[0]])
        x = np.linspace(0, size, size)
        fig.add_trace(go.Scatter(x=x, y=cum_mean(data_test[group]),
                    mode="lines",
                    name='group: {}'.format(group)))

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=True,
        ),
        autosize=True,
        showlegend=True,
        plot_bgcolor='white'
    )
    return fig

def groups_option_ab_test(data_test):
    """
    define the set of groups that can be selected by the user
    Args:
        data_test: dataframe with the data of the test
    Return:
        list of dict: list of dict for each group
    """
    groups_option = []
    for i in data_test.columns:
        dict = {'label': 'groupe {}'.format(i), 'value': i}
        groups_option.append(dict)
    return groups_option


def heatmap_plotly_bayesian_test(results_a, results_b, name_a="A", name_b="B", steps=100):
    """
    plot the heatmap of joint probability for the paramaterters of two groups
    Arg:
        results_a: data for group A
        results_b: data for group B
    Return:
        fig plotly
    """
    posterior_law_a, posterior_law_b = compute_posterior_laws(results_a, results_b)
    posterior_a, posterior_b = compute_posterior_probs(posterior_law_a, posterior_law_b, steps)
    posterior_joint_probs = compute_joint_posterior_probs(posterior_a, posterior_b)

    fig = go.Figure(data=go.Heatmap(
                        x=[i / steps for i in range(steps)],
                        y=[i / steps for i in range(steps)],
                        z=posterior_joint_probs))

    fig.update_layout(xaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showline=False
                                ),
                      yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showline=False,
                            showticklabels=True,
                                ),
                     autosize=False)

    fig.update_layout(
        autosize=True,
        width=500,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
        paper_bgcolor="white",
    )
    fig.update_xaxes(title_text='parameter value for group {}'.format(name_b))
    fig.update_yaxes(title_text='parameter value for group {}'.format(name_a))
    
    return fig