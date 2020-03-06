# -*- coding: utf-8 -*-


import os
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler
import mpl_toolkits.mplot3d
import numpy as np
from tools import *
from scipy.stats import rankdata
import seaborn as sns
import pandas as pd
sns.set(font_scale=0.5)

# READ RAW DATA FILES
def read_data_file(data_file='data_files/brute-force-data.json'):
    with open(data_file, 'r') as json_file:
        data = json.load(json_file)

    # Return only the data variable
    return data


# Read the data files
data_frame = read_data_file()

# %%
folder_name = 'data_files/images/'
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)


# problems = list(set(data_frame['problem']))
problems = [data_frame['problem'][index] for index in sorted(
    np.unique(data_frame['problem'], return_index=True)[1])]
operators = (data_frame['results'][0]['operator_id'])
dimensions = sorted(list(set(data_frame['dimensions'])))

# Show the variable tree
printmsk(data_frame)

# %% PLOT FITNESS PER CARD/DIMENSION
is_saving = False

# Special adjustments
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=6)

# Initialise variables
# Min=list(), Med=list(), Avg=list(), Std=list(),
             # Max=list(), IQR=list(), MAD=list())

# Obtain matrices for problems and operators
operator_matrix, problem_matrix = np.meshgrid(
    np.array(operators), np.arange(len(problems)))

# Plot a figure per dimension
for dimension in dimensions:
    # Find indices corresponding to such a dimension
    dim_indices = listfind(data_frame['dimensions'], dimension)

    # Get fitness statistical values and stored them in 'arrays'
    temporal_array = list()

    # Selecting data
    for dim_index in dim_indices:  # Problems
        temporal_dict = data_frame['results'][dim_index]['statistics']
        ranked_data = rankdata([
            temporal_dict[op_index]['Min'] + temporal_dict[op_index]['IQR']
            for op_index in range(len(operators))])
        temporal_array.append(ranked_data)
    stats = pd.DataFrame(temporal_array, index=problems, columns=operators)

    # Printing section
    fig = plt.figure(figsize=(11, 8.5), dpi=333)

    # ax = fig.gca(projection='3d')
    # ax.plot_surface(operator_matrix, problem_matrix, stats[key])
    ax = sns.heatmap(stats, cbar=False, cmap=plt.cm.gray)  # , vmin=1, vmax=5)
    # fig.tight_layout()
    plt.title("Dim: {}".format(dimension))
    plt.show()

"""

Multiple 3D Surface Plots

import plotly.graph_objects as go
import numpy as np

z1 = np.array([
    [8.83,8.89,8.81,8.87,8.9,8.87],
    [8.89,8.94,8.85,8.94,8.96,8.92],
    [8.84,8.9,8.82,8.92,8.93,8.91],
    [8.79,8.85,8.79,8.9,8.94,8.92],
    [8.79,8.88,8.81,8.9,8.95,8.92],
    [8.8,8.82,8.78,8.91,8.94,8.92],
    [8.75,8.78,8.77,8.91,8.95,8.92],
    [8.8,8.8,8.77,8.91,8.95,8.94],
    [8.74,8.81,8.76,8.93,8.98,8.99],
    [8.89,8.99,8.92,9.1,9.13,9.11],
    [8.97,8.97,8.91,9.09,9.11,9.11],
    [9.04,9.08,9.05,9.25,9.28,9.27],
    [9,9.01,9,9.2,9.23,9.2],
    [8.99,8.99,8.98,9.18,9.2,9.19],
    [8.93,8.97,8.97,9.18,9.2,9.18]
])

z2 = z1 + 1
z3 = z1 - 1

fig = go.Figure(data=[
    go.Surface(z=z1),
    go.Surface(z=z2, showscale=False, opacity=0.9),
    go.Surface(z=z3, showscale=False, opacity=0.9)

])

fig.show()

"""

