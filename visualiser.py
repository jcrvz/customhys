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

problem_features_without_code = bf.list_functions()
# for key in problem_features_without_code.keys():
#     del problem_features_without_code[key]['Code']
categorical_features = pd.DataFrame(problem_features_without_code).T
categories =  categorical_features.groupby('Code').first()
categories['Members'] = categorical_features.groupby('Code').count().mean(axis=1)
# simple_problem_list = sorted([[x['Name'], x['Code']] for x in problem_features], key=lambda y: y[1])


# READ RAW DATA FILES
def read_data_file(data_file='data_files/brute-force-data.json'):
    with open(data_file, 'r') as json_file:
        data = json.load(json_file)

    # Return only the data variable
    return data


# Read heuristic space
with open('collections/' + 'default.txt', 'r') as operators_file:
    heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]
search_operators = [x[0].replace('_', ' ') + "-" +
                    "-".join(["{}".format(y) for y in [*x[1].values()]]).replace('_', ' ') + "-" +
                    x[2] for x in heuristic_space]

# Read the data files
data_frame = read_data_file()

# Show the variable tree
# printmsk(data_frame)

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
operator_matrix, problem_matrix = np.meshgrid(np.array(operators), np.arange(len(problems)))

# Naive weights for tests
operators_weights = dict()

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

    # Create the data frame
    stats = pd.DataFrame(temporal_array, index=problems, columns=operators)
    stats['Group'] = pd.Series(problems_weights, index=stats.index)

    # -- PART 1: PLOT THE CORRESPONDING HEATMAP --
    # Delete the Group column
    stats_without_category = stats.sort_values(by=['Group']).drop(columns=['Group'])

    # Printing section
    fig = plt.figure(figsize=(15, 10), dpi=333)

    ax = sns.heatmap(stats_without_category, cbar=False, cmap=plt.cm.rainbow,
                     robust=True, xticklabels=True, yticklabels=True)  # , vmin=1, vmax=5)

    plt.title("Dim: {}".format(dimension))
    plt.show()

    if is_saving:
        fig.savefig(folder_name + 'raw-heatmap-bruteforce-{}D'.format(dimension) + '.pdf',
                    format='pdf', dpi=fig.dpi)

    # -- PART 2: OBTAIN NAIVE INSIGHTS
    grouped_stats = stats.groupby('Group').mean()
    prop_stats = grouped_stats.div(grouped_stats.sum(axis=1), axis=0)

    # Store weights in the final dictionaru
    operators_weights[dimension] = df2dict(prop_stats)

    fig = plt.figure(figsize=(0.08*205, 0.4*8), dpi=333)
    ax = sns.heatmap(prop_stats, cbar=False, cmap=plt.cm.rainbow,
                     robust=True, xticklabels=True, yticklabels=True)
    plt.title("Dim: {}".format(dimension))
    plt.yticks(rotation=0)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    if is_saving:
        fig.savefig(folder_name + 'cat-heatmap-bruteforce-{}D'.format(dimension) + '.pdf',
                    format='pdf', dpi=fig.dpi)

save_json(operators_weights, 'operators_weights')