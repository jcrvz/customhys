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
import benchmark_func as bf
sns.set(font_scale=0.5)


# Read benchmark functions and their features
problem_features = bf.list_functions()
# simple_problem_list = sorted([[x['Name'], x['Code']] for x in problem_features], key=lambda y: y[1])


# READ RAW DATA FILES
def read_data_file(data_file='data_files/brute-force-data.json'):
    with open(data_file, 'r') as json_file:
        data = json.load(json_file)

    # Return only the data variable
    return data


# Read the data files
data_frame = read_data_file()

# Show the variable tree
printmsk(data_frame)

# %%
folder_name = 'data_files/images/'
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

# problems = list(set(data_frame['problem']))
problems = [data_frame['problem'][index] for index in sorted(np.unique(data_frame['problem'], return_index=True)[1])]
problems_weights = [problem_features[x]['Code'] for x in problems]
operators = (data_frame['results'][0]['operator_id'])
dimensions = sorted(list(set(data_frame['dimensions'])))

# %% PLOT FITNESS PER CARD/DIMENSION
is_saving = True

# Special adjustments
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=4)

# Initialise variables
# Min=list(), Med=list(), Avg=list(), Std=list(),
             # Max=list(), IQR=list(), MAD=list())
# number_bins = 51
# bins = np.ceil(np.logspace(-1, 2, number_bins))

# Obtain matrices for problems and operators
operator_matrix, problem_matrix = np.meshgrid(
    np.array(operators), np.arange(len(problems)))

# Plot a figure per dimension
for dimension in dimensions:
    # Find indices corresponding to such a dimension
    dim_indices = listfind(data_frame['dimensions'], dimension)

    # Get fitness statistical values and stored them in 'arrays'
    temporal_array = list()
    problem_categories = list()

    # Selecting data
    for dim_index in dim_indices:  # Problems
        temporal_dict = data_frame['results'][dim_index]['statistics']
        problem_categories.append(problem_features[data_frame['problem'][dim_index]]['Code'])
        ranked_data = rankdata([
            temporal_dict[op_index]['Min'] + temporal_dict[op_index]['IQR']
            for op_index in range(len(operators))])
        temporal_array.append(ranked_data)

    # bins = np.linspace(np.min(temporal_array), np.max(temporal_array), number_bins)
    # temporal_array = np.digitize(temporal_array, bins)

    stats = pd.DataFrame(temporal_array, index=problems, columns=operators)
    stats['Group'] = pd.Series(problems_weights, index=stats.index)
    stats = stats.sort_values(by=['Group']).drop(columns=['Group'])
    # stats = stats.sort_values(by=['Group'])  # .drop(columns=['Group'])

    # Printing section
    fig = plt.figure(figsize=(15, 10), dpi=333)

    # ax = fig.gca(projection='3d')
    # ax.plot_surface(operator_matrix, problem_matrix, stats[key])
    ax = sns.heatmap(stats, cbar=False, cmap=plt.cm.rainbow, robust=True, xticklabels=True, yticklabels=True)  # , vmin=1, vmax=5)
    # ax = sns.clustermap(stats, cmap=plt.cm.rainbow, cbar=False, col_cluster=True)


    # fig.tight_layout()
    plt.title("Dim: {}".format(dimension))
    plt.show()

    # grouped = stats.groupby(by=['Group'])
    # ncols = 1
    # nrows = int(np.ceil(grouped.ngroups / ncols))
    #
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
    #
    # for (key, ax1) in zip(grouped.groups.keys(), axes.flatten()):
    #     sns.heatmap(grouped.get_group(key), cbar=False, cmap=plt.cm.rainbow, robust=True, ax=ax1)

    if is_saving:
        fig.savefig(folder_name + 'heatmap-bruteforce-{}D'.format(dimension) + '.pdf',
                    format='pdf', dpi=fig.dpi)

    # break

