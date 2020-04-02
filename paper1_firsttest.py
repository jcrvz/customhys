'''
To run this file you must need:
i. A folder "data_files" containing "first_test.json"  which is created by using
    tools.preprocess_first_files
ii. A collection of heuristics called "default.txt" which was handmade because the
    hyperparameter values it uses.

@authors:   Jorge Mario Cruz-Duarte (jcrvz.github.io)
'''


# Load packages
import os
import matplotlib.pyplot as plt
import numpy as np
import tools as jt
from scipy.stats import rankdata
# import seaborn as sns
import pandas as pd
import benchmark_func as bf
from scipy import stats
# sns.set(font_scale=0.5)

# Read benchmark functions and their features
problem_features = bf.list_functions()
problem_names = bf.for_all('func_name')

# Read benchmark functions, their features and categorise them
problem_features_without_code = bf.list_functions()

# Read heuristic space
with open('collections/' + 'default.txt', 'r') as operators_file:
    heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]
search_operators = [x[0].replace('_', ' ') + "-" + "-".join(["{}".format(y) for y in [
    *x[1].values()]]).replace('_', ' ') + "-" + x[2] for x in heuristic_space]

# Read the data files
data_frame = jt.read_json('data_files/first_test.json')

# Show the variable tree
# printmsk(data_frame)

# Prepare additional lists (like problems, weights, operators, and dimensions)
problems = [data_frame['problem'][index] for index in sorted(np.unique(data_frame['problem'], return_index=True)[1])]

dimensions = sorted(list(set(data_frame['dimensions'])))

# Saving images flag
is_saving = False

folder_name = 'data_files/images/'
if is_saving:
    # Read (of create if so) a folder for storing images
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


# %% PLOT LAST-PERFORMANCE (FOUND METAHEURISTIC) PER DIMENSION

# Special adjustments for the plots
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=4)

# Plot a figure per dimension
for dimension in dimensions:
    # Find indices corresponding to such a dimension
    dim_indices = jt.listfind(data_frame['dimensions'], dimension)

    # Get fitness statistical values and stored them in 'arrays'
    steps = list()
    performances = list()
    solutions = list()
    fitness_values = list()
    problem = list()
    problem_categories = list()
    statistics = list()
    pvalues = list()

    # Selecting data
    for dim_index in dim_indices:  # Problems
        current_problem = problem_names[data_frame['problem'][dim_index]]
        # current_optimum = problem_optima[current_problem][0]

        steps.append(data_frame['results'][dim_index]['step'])
        performances.append(data_frame['results'][dim_index]['performance'])
        solutions.append(data_frame['results'][dim_index]['encoded_solution'])

        last_historical_fitness = [x[-1] for x in data_frame['results'][dim_index]['hist_fitness']]
        fitness_values.append(last_historical_fitness)
        statistics.append(data_frame['results'][dim_index]['statistics'])

        # Normality test
        _, pvalue = stats.normaltest(last_historical_fitness)
        pvalues.append(pvalue)

        problem.append(current_problem)
        problem_categories.append(problem_features[data_frame['problem'][dim_index]]['Code'])

    # -- PART 1: PLOT THE CORRESPONDING HEATMAP --
    # Delete the Group column
    # stats_without_category = stats.sort_values(by=['Group']).drop(columns=['Group'])
    data_per_dimension = pd.DataFrame({
        'Problem': problem,
        'Category': problem_categories,
        'p-Value': pvalues,
        'Performance': [x[-1] for x in performances],
        'Median-Fitness': [x[-1]['Med'] for x in statistics],
        'IQR-Fitness': [x[-1]['IQR'] for x in statistics],
        'Avg-Fitness': [x[-1]['Avg'] for x in statistics],
        'Std-Fitness': [x[-1]['Std'] for x in statistics],
        'Metaheuristic': [x[-1] for x in solutions]
    }).sort_values(by=['Category', 'Problem'])

    # data_per_dimension.to_csv('data_files/first_pd{}D.csv'.format(dimension), index=False)
    with open('data_files/first_pd{}D.tex'.format(dimension), 'w') as tf:
        tf.write(data_per_dimension.to_latex(index=False, header=[
            'Problem', 'Category', 'p-value', 'Performance', 'Median', 'IQR', 'Avg.', 'St. Dev.', 'MH indices'
        ]))

    # if is_saving:
    #     fig.savefig(folder_name + 'raw-heatmap-firsttest-{}D'.format(dimension) + '.pdf', format='pdf', dpi=fig.dpi)


