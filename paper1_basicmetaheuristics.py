'''
To run this file you must need:
i. A folder "data_files" containing "brute-force-data.json"  which is created by using
    tools.preprocess_bruteforce_files.
ii. A collection of heuristics called "default.txt" which was handmade because the
    hyperparameter values it uses.

@authors:   Jorge Mario Cruz-Duarte (jcrvz.github.io)
'''


# Load packages
import os
import matplotlib.pyplot as plt
import numpy as np
import tools as jt
from scipy.stats import rankdata, mode
import seaborn as sns
import pandas as pd
import benchmark_func as bf
sns.set(font_scale=0.5)

# Read benchmark functions and their features
# problems = bf.__all__
problem_features = bf.list_functions()

# Read benchmark functions, their features and categorise them
problem_features_without_code = bf.list_functions()
categorical_features = pd.DataFrame(problem_features_without_code).T
categories = categorical_features.groupby('Code').first()
categories['Members'] = categorical_features.groupby('Code').count().mean(axis=1)

# Read the data files
data_frame = jt.read_json('data_files/basic-metaheuristics-data.json')

# Show the variable tree
# printmsk(data_frame)

# Prepare additional lists (like problems, weights, operators, and dimensions)
problems = [data_frame['problem'][index] for index in sorted(np.unique(data_frame['problem'], return_index=True)[1])]

problems_categories = {x: problem_features[x]['Code'] for x in problems}
operators = (data_frame['results'][0]['operator_id'])
dimensions = sorted(list(set(data_frame['dimensions'])))

# Naive weights for tests
metaheuristics_modes = dict()

# Saving images flag
is_saving = True

folder_name = 'data_files/images/'
if is_saving:
    # Read (of create if so) a folder for storing images
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

# %% PLOT FITNESS PER CARD/DIMENSION

# Special adjustments for the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=4)

# Obtain matrices for problems and operators
operator_matrix, problem_matrix = np.meshgrid(np.array(operators), np.arange(len(problems)))

# 1st metaheuristic per problem+dimension
best_metaheuristics = {prob: [] for prob in problems}

# Plot a figure per dimension
for dimension in dimensions:
    # Find indices corresponding to such a dimension
    dim_indices = jt.listfind(data_frame['dimensions'], dimension)

    # Get fitness statistical values and stored them in 'arrays'
    temporal_array = list()
    problem_categories = list()
    first_metaheuristics = []

    # Selecting data
    for dim_index in dim_indices:  # Problems
        temporal_dict = data_frame['results'][dim_index]['statistics']
        problem_categories.append(problem_features[data_frame['problem'][dim_index]]['Code'])
        ranked_data = rankdata([temporal_dict[op_index]['Min'] + temporal_dict[op_index]['IQR']
                                for op_index in range(len(operators))])
        temporal_array.append(ranked_data)

        # Save only the best one
        first_metaheuristics.append(np.argsort(ranked_data)[0])
        best_metaheuristics[data_frame['problem'][dim_index]].append(np.argsort(ranked_data)[0])

    # Create the data frames
    stats = pd.DataFrame(temporal_array, index=problems, columns=operators)
    stats['Group'] = pd.Series(problems_categories, index=stats.index)
    stats_without_category = stats.sort_values(by=['Group']).drop(columns=['Group'])

    best_mhs = pd.DataFrame(first_metaheuristics, index=problems, columns=['BestMH'])
    best_mhs['Group'] = pd.Series(problems_categories, index=best_mhs.index)
    grouped_stats = best_mhs.groupby('Group').apply(pd.DataFrame.mode).drop(columns=['Group'])
    grouped_stats.index = [x[0] for x in grouped_stats.index]

    metaheuristics_modes[dimension] = jt.df2dict(grouped_stats)

    # -- PART 1: PLOT THE CORRESPONDING HEATMAP --
    # Delete the Group column
    # stats_without_category = stats.sort_values(by=['Group']).drop(columns=['Group'])

    # Printing section
    fig = plt.figure(figsize=(15, 10), dpi=333)

    ax = sns.heatmap(stats_without_category, cbar=False, cmap='rainbow', robust=True,
                     xticklabels=True, yticklabels=True)  # , vmin=1, vmax=5)

    plt.title("Dim: {}".format(dimension))
    plt.show()

    if is_saving:
        fig.savefig(folder_name + 'raw-heatmap-basicmetaheuristic-{}D'.format(dimension) + '.pdf', format='pdf', dpi=fig.dpi)

# Save weight for the benchmark problems contained in brute-force-data.json and default.txt
for key, val in best_metaheuristics.items():
    best_metaheuristics[key] = np.array(val)
jt.save_json(best_metaheuristics, 'data_files/best_basicmetaheuristics')
jt.save_json(metaheuristics_modes, 'data_files/modes_metaheuristics')