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
data_frame = jt.read_json('data_files/first_test_v1.json')  # up to 3 with 100 iterations
# data_frame = jt.read_json('data_files/first_test_v2.json')  # up to 5 with 100 iterations

# Load results from basic metaheuristics
data_basics = jt.read_json('data_files/basic-metaheuristics-data.json')
basic_metaheuristics = data_basics['results'][0]['operator_id']

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
    mh_performances = list()
    solutions = list()
    mh_ids = list()
    fitness_values = list()
    problem = list()
    problem_categories = list()
    statistics = list()
    pvalues = list()
    new_vs_basic = list()

    # Selecting data
    for dim_index in dim_indices:  # Problems
        # Read the problems that correspond to this dimension
        current_problem = problem_names[data_frame['problem'][dim_index]]
        # current_optimum = problem_optima[current_problem][0]

        # Read the steps, performances, and solutions for each problem
        steps.append(data_frame['results'][dim_index]['step'])
        # current_performance = data_frame['results'][dim_index]['performance']
        current_performance = [x['Med'] + x['IQR'] + x['Avg'] for x in data_frame['results'][dim_index]['statistics']]
        performances.append(current_performance)
        solutions.append(data_frame['results'][dim_index]['encoded_solution'])

        # Read the last historical fitness evolution (it is for illustrative purposes)
        last_historical_fitness = [x[-1] for x in data_frame['results'][dim_index]['hist_fitness']]
        fitness_values.append(last_historical_fitness)
        statistics.append(data_frame['results'][dim_index]['statistics'])

        # Normality test of the obtained results
        _, pvalue = stats.normaltest(last_historical_fitness)
        pvalues.append(pvalue)

        # Read the performance reached by all the best basic metaheuristic for this problem/dim
        # performance_basics = data_basics['results'][dim_index]['performance']
        temporal_dict = data_basics['results'][dim_index]['statistics']
        performance_basics = [temporal_dict[op_index]['Med'] + temporal_dict[op_index]['IQR'] +
                              temporal_dict[op_index]['Avg']
                              for op_index in range(len(basic_metaheuristics))]
        min_mh_performance = np.min(performance_basics)
        mh_performances.append(min_mh_performance)
        mh_ids.append(np.argmin(performance_basics))
        new_vs_basic.append(current_performance[-1] - min_mh_performance)

        problem.append(current_problem)
        problem_categories.append(problem_features[data_frame['problem'][dim_index]]['Code'])

    # Generate a dataframe to plot the results
    data_per_dimension = pd.DataFrame({
        'Problem': problem,
        'Category': problem_categories,
        'p-Value': pvalues,
        'Performance': [x[-1] for x in performances],
        'Median-Fitness': [x[-1]['Med'] for x in statistics],
        'IQR-Fitness': [x[-1]['IQR'] for x in statistics],
        'Avg-Fitness': [x[-1]['Avg'] for x in statistics],
        'Std-Fitness': [x[-1]['Std'] for x in statistics],
        'Metaheuristic': [x[-1] for x in solutions],
        'BasicMH_Performance': mh_performances,
        'BasicMH_Ids': mh_ids,
        'perfNew-Basic': new_vs_basic
    }).sort_values(by=['Category', 'Problem'])

    # -- PART 1: PLOT THE CORRESPONDING VIOLIN PLOTS --
    # Delete the Group column
    # stats_without_category = stats.sort_values(by=['Group']).drop(columns=['Group'])

    # data_per_dimension.to_csv('data_files/first_pd{}D.csv'.format(dimension), index=False)
    # with open('data_files/first_pd{}D.tex'.format(dimension), 'w') as tf:
    #     tf.write(data_per_dimension.to_latex(index=False, header=[
    #         'Problem', 'Category', 'p-value', 'Performance', 'Median', 'IQR', 'Avg.',
    #         'St. Dev.', 'MH indices', 'Basic MH', 'Basic MH Ind', 'Perf. MH-Basic'
    #     ]))

    y_data = np.array(new_vs_basic) < 0.0
    x_data = np.arange(len(y_data))
    plt.plot(x_data, y_data, '-o')
    plt.title("Dim: {}, {}/{} = {:.2f}%".format(dimension, np.sum(y_data), len(y_data),
                                                100 * np.sum(y_data)/len(y_data)))
    plt.show()

    # if is_saving:
    #     fig.savefig(folder_name + 'raw-heatmap-firsttest-{}D'.format(dimension) + '.pdf', format='pdf', dpi=fig.dpi)


