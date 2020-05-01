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
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d
import numpy as np
import tools as jt
from scipy.stats import rankdata
# import seaborn as sns
import pandas as pd
import benchmark_func as bf
from scipy import stats
# sns.set(context="paper", font_scale=1, palette="colorblind", style="ticks")

# Read benchmark functions and their features
problem_features = bf.list_functions()
problem_names = bf.for_all('func_name')

# DEL: Read benchmark functions, their features and categorise them
# problem_features_without_code = bf.list_functions()

# %% CASES

# Set the test case
test_case = 3

# Saving images flag
is_saving = True

# Choose the corresponding case
if test_case == 1:
    # Problem collection
    operators_collection = 'default'

    # Read data from new (tailored) metaheuristics
    new_mhs_data = jt.read_json('data_files/first_test_v1.json');
    card_upto = 3  # up to 3 with 100 iterations

    # Saving label
    saving_label = 'short1'

elif test_case == 2:
    # Problem collection
    operators_collection = 'default'

    # Read data from new (tailored) metaheuristics
    new_mhs_data = jt.read_json('data_files/first_test_v2.json');
    card_upto = 5  # up to 5 with 100 iterations

    # Saving label
    saving_label = 'short2'

elif test_case == 3:
    # Problem collection
    operators_collection = 'test-set-21'

    # Read data from new (tailored) metaheuristics
    new_mhs_data = jt.read_json('data_files/first_test_v3.json');
    card_upto = 3  # up to 3 with 100 iterations

    # Saving label
    saving_label = 'long1'

# %%

# Read search operators
with open('collections/' + operators_collection + '.txt', 'r') as operators_file:
    heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]

# Process search operators as strings
search_operators = [x[0].replace('_', ' ') + ", PAR: (" +
                    ", ".join(["{}".format(y) for y in [*x[1].values()]]).replace('_', ' ') +
                    "), SEL: " + x[2] for x in heuristic_space]

# Read basic metaheuristics
with open('collections/' + 'basicmetaheuristics.txt', 'r') as operators_file:
    basic_mhs_collection = [eval(line.rstrip('\n')) for line in operators_file]


def read_cardinality(x):
    if isinstance(x, tuple):
        return 1
    if isinstance(x, list):
        return len(x)


# Read basic metaheuristics cardinality
basic_mhs_cadinality = [read_cardinality(x) for x in basic_mhs_collection]

# %%

# Load data from basic metaheuristics
basic_mhs_data = jt.read_json('data_files/basic-metaheuristics-data.json')
basic_metaheuristics = basic_mhs_data['results'][0]['operator_id']

# %%

# Show the variable tree
# printmsk(new_mhs_data)

# Read the dimensions executed
dimensions = sorted(list(set(new_mhs_data['dimensions'])))

# Data Frames per dimensions
data_per_dimension = list()

# Check if the image folder exists
folder_name = 'data_files/images/'
if is_saving:
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

# %%

# Special adjustments for the plots, i.e., TeX fonts and so on
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
plt.rc('axes', facecolor='white')
success_per_category = dict(mean=list(), std=list(), median=list(), q1=list(), q3=list())

# %% For each dimension do ...
for dimension in dimensions:

    # %% dimension = 2

    # Find indices corresponding to such a dimension
    dim_indices = jt.listfind(new_mhs_data['dimensions'], dimension)

    # Get fitness statistical values and stored them in 'arrays'
    steps = list()
    performances = list()
    best_mh_performance = list()
    solutions = list()
    best_mh_index = list()
    fitness_values = list()
    problem_string = list()
    problem_categories = list()
    statistics = list()
    pvalues = list()
    new_vs_basic = list()
    success_rate = list()

    # %%

    # For each problem
    for problem_index in dim_indices:

        # %% problem_index = 0

        # Read the problem name that correspond to this dimension
        current_problem = problem_names[new_mhs_data['problem'][problem_index]]

        # Store problem name
        problem_string.append(current_problem)

        # Store problem category
        problem_categories.append(problem_features[new_mhs_data['problem'][problem_index]]['Code'])

        # Store the number of last step
        steps.append(new_mhs_data['results'][problem_index]['step'][-1])

        # Read the current performance
        current_performance = np.copy(new_mhs_data['results'][problem_index]['performance'][-1])

        # Store this performance
        performances.append(current_performance)

        # Store the corresponding solution
        solutions.append(new_mhs_data['results'][problem_index]['encoded_solution'][-1])

        # Read the last historical fitness values (for illustrative purposes)
        last_historical_fitness = [x[-1] for x in new_mhs_data['results'][problem_index]['hist_fitness']]

        # Store the last historical fitness
        fitness_values.append(last_historical_fitness)

        # Store the statistics
        statistics.append(new_mhs_data['results'][problem_index]['statistics'][-1])

        # Perform normality test from historical fitness data
        _, pvalue = stats.normaltest(last_historical_fitness)

        # Store the p-value
        pvalues.append(pvalue)

        # Read the performance of basic metaheuristics
        basic_performances = np.copy(basic_mhs_data['results'][problem_index]['performance'])

        # %% Compare tailored metaheuristic with basic metaheuristics

        # Compare the current metaheuristic against the basic metaheuristics
        performance_comparison = np.copy(current_performance - np.array(basic_performances))

        # Success rate with respect to basic metaheuristics
        success_rate.append(np.sum(performance_comparison < 0.0) / len(performance_comparison))

        # Find the best basic metaheuristic and its performance
        min_mh_performance = np.min(basic_performances)
        best_mh_performance.append(min_mh_performance)
        best_mh_index.append(np.argmin(basic_performances))

        # Store the comparison between the current metaheuristic with the best basic metaheuristic
        new_vs_basic.append(current_performance - min_mh_performance)

    # %% Store all the previous information in a DataFrame

    # Generate a dataframe to plot the results
    current_data_per_dimension = pd.DataFrame({
        'Problem': problem_string,
        'Category': problem_categories,
        'p-Value': pvalues,
        'Performance': performances,
        'Median-Fitness': [x['Med'] for x in statistics],
        'IQR-Fitness': [x['IQR'] for x in statistics],
        'Avg-Fitness': [x['Avg'] for x in statistics],
        'Std-Fitness': [x['Std'] for x in statistics],
        'Metaheuristic': solutions,
        'BasicMH Performance': best_mh_performance,
        'BasicMH Ids': best_mh_index,
        'perfNew-Basic': new_vs_basic,
        'success-Rate': success_rate
    }).sort_values(by=['Category', 'Problem'])
    data_per_dimension.append(current_data_per_dimension)

    # data_per_dimension.to_csv('data_files/first_pd{}D.csv'.format(dimension), index=False)
    # with open('data_files/first_pd{}D.tex'.format(dimension), 'w') as tf:
    #     tf.write(data_per_dimension.to_latex(index=False, header=[
    #         'Problem', 'Category', 'p-value', 'Performance', 'Median', 'IQR', 'Avg.',
    #         'St. Dev.', 'MH indices', 'Basic MH', 'Basic MH Ind', 'Perf. MH-Basic', 'Success-Rate'
    #     ]))

    # %% Obtain the success rate per category
    # success_per_category.append([*data_per_dimension.groupby("Category")["perfNew-Basic"].agg(
    #     lambda x: np.sum(np.array(x) < 0.0) / len(x)).values])
    success_per_category['mean'].append(
        [*current_data_per_dimension.groupby("Category")["success-Rate"].agg('mean').values])
    success_per_category['std'].append(
        [*current_data_per_dimension.groupby("Category")["success-Rate"].agg(np.std).values])
    success_per_category['median'].append(
        [*current_data_per_dimension.groupby("Category")["success-Rate"].agg('median').values])
    success_per_category['q1'].append(
        [*current_data_per_dimension.groupby("Category")["success-Rate"].agg(lambda x: np.quantile(x, 0.25)).values])
    success_per_category['q3'].append(
        [*current_data_per_dimension.groupby("Category")["success-Rate"].agg(lambda x: np.quantile(x, 0.75)).values])

    # %% FIRST PLOT: Success rate per problem

fig = plt.figure(figsize=(4, 3), dpi=125, facecolor='w')
ax = fig.gca(projection='3d', proj_type='ortho', azim=140, elev=40)
plt.ion()

x1 = np.arange(len(success_rate)) + 1

for dim_ind in range(len(dimensions)):
    y1 = dim_ind * np.ones(len(success_rate))
    z1 = data_per_dimension[dim_ind]['success-Rate']

    ax.plot3D(x1, y1, z1)

plt.ioff()

plt.yticks(np.arange(len(y1)), dimensions)
plt.ylim(0, len(dimensions)-1)
plt.xlim(1, x1[-1] + 1 )
ax.set_xticks(np.round(np.linspace(1, 107, 7)))
ax.set_ylabel(r'Dimension')
ax.set_xlabel(r'Problem Id.')
ax.set_zlabel(r'Success Rate')
plt.tight_layout()

if is_saving:
    plt.savefig(folder_name + 'Exp-SuccessRatePerDimFunc_{}.svg'.format(saving_label), format='svg',
                facecolor=fig.get_facecolor(), dpi=333)

plt.show()

# %% SUCCESS RATE FOR ALL DIMENSIONS PER CATEGORY
categories = [*data_per_dimension[0].groupby("Category")["success-Rate"].agg('mean').index.values]

fig = plt.figure(figsize=(5, 3), dpi=125, facecolor='w')
plt.ion()

y0 = np.array(success_per_category['mean'])
y1 = y0 - np.array(success_per_category['std'])
y2 = y1 + np.array(success_per_category['std'])

cmap = plt.get_cmap('tab10')
colors = [cmap(i)[:-1] for i in np.linspace(0, 1, len(categories))]


for k, color in enumerate(colors, start=0):
    plt.fill_between(np.arange(len(dimensions)), y1[:, k], y2[:, k], color=color, alpha=0.1)
    plt.plot(np.arange(len(dimensions)), y0[:, k], color=color)

# plt.plot(dimensions, np.array(success_per_category['mean']), '--')
# plt.xticks(range(len(dimensions)), dimensions)
plt.ioff()
plt.legend(categories, frameon=False, loc='lower right', ncol=2)
plt.xlabel(r'Dimensions')
plt.ylabel(r'Success Rate')
plt.ylim((0.0, 1))
plt.xlim(0, len(dimensions)-1)
plt.xticks(np.arange(len(dimensions)), dimensions)
plt.tight_layout()

if is_saving:
    plt.savefig(folder_name + 'Exp-SuccessRatePerDimCat_{}.pdf'.format(saving_label), format='pdf', dpi=333)

plt.show()

# %% SUCCESS RATE FOR ALL DIMENSIONS TOTAL

# fig = plt.figure(figsize=(4, 3), dpi=125, facecolor='w')
# y_data = np.array([x['success-Rate'].values for x in data_per_dimension])
# x_data = np.arange(len(dimensions))
#
# violin_parts = plt.violinplot(y_data.T, x_data,
#         showmeans=True, showmedians=True, showextrema=False)
# plt.xticks(x_data, labels=dimensions)
#
# violin_parts['cmeans'].set_edgecolor('#AC4C3D')  # Rojo
# violin_parts['cmeans'].set_linewidth(1.5)
#
# violin_parts['cmedians'].set_edgecolor('#285C6B')  # Azul
# violin_parts['cmedians'].set_linewidth(1.5)
#
# for vp in violin_parts['bodies']:
#     vp.set_edgecolor('#523069')  # Moradito oscuro
#     vp.set_facecolor('#A149C1')  # Moradito suave
#     vp.set_linewidth(1.0)
#     vp.set_alpha(0.5)
#
# plt.ylabel(r'Success Rate')
# plt.xlabel(r'Dimensions')
# plt.ioff()
# plt.legend([Line2D([0], [0], color='#AC4C3D', lw=3),
#             Line2D([0], [0], color='#285C6B', lw=3)],
#            ['Mean', 'Median'], frameon=False, loc='lower right')
# plt.ylim(-0.01, 1.01)
# plt.xlim(x_data[0], x_data[-1])
# plt.tight_layout()
#
# if is_saving:
#     plt.savefig(folder_name + 'Exp-SuccessRatePerDim_{}.pdf'.format(saving_label), format='pdf', dpi=333)
#
# plt.show()

# %% CARDINALITY FOR ALL DIMENSIONS TOTAL

# fig = plt.figure(figsize=(6, 3), dpi=125, facecolor='w')
# y_data = np.array([[len(y) for y in x['Metaheuristic'].values] for x in data_per_dimension])

#
# plt.hist(y_data.T, [*(card_bin_centres - .5), card_bin_centres[-1] + .5], density=True, histtype='bar')
# plt.xticks(card_bin_centres)
#
# plt.ylabel(r'Normalised Frequency')
# plt.xlabel(r'Cardinality')
# plt.ioff()
# plt.legend(dimensions, frameon=False, loc='upper left')
# plt.ylim(0, 0.8)
# plt.tight_layout()
#
# if is_saving:
#     plt.savefig(folder_name + 'Exp-CardinalityPerDim_{}.pdf'.format(saving_label), format='pdf', dpi=333)
#
# plt.show()

# %% CARDINALITY FOR ALL DIMENSIONS TOTAL (other view)

fig = plt.figure(figsize=(6, 3), dpi=125, facecolor='w')
y_data = np.array([[len(y) for y in x['Metaheuristic'].values] for x in data_per_dimension])
card_bin_centres = np.arange(np.max(y_data)) + 1
x_data = np.arange(len(dimensions))

plt.ion()

num_cards = len(card_bin_centres)
bin_width = 1/(num_cards + 1)
sub_bins = np.linspace(-0.5+bin_width, 0.5-bin_width, num_cards)

colours = plt.cm.tab10(np.linspace(0, 1, 10))

#  ---
# N = 20
# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = 10 * np.random.rand(N)
# width = np.pi / 4 * np.random.rand(N)
# colors = plt.cm.viridis(radii / 10.)
#
# ax = plt.subplot(111, projection='polar')
# ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)
#  --

for k in range(len(dimensions)):
    hist_data = np.histogram(y_data[k,:], [*(card_bin_centres - .5), card_bin_centres[-1] + .5], density=True)[0]
    for i in range(num_cards):
        plt.bar(k + sub_bins[i], hist_data[i], color=colours[i, :], width=bin_width, align='center')
        # bottom=np.sum(hist_data[:i]),

plt.xticks(x_data)

plt.ylabel(r'Normalised Frequency')
plt.xlabel(r'Dimensions')
plt.xticks(x_data, dimensions)
plt.ioff()
plt.legend([r'{}'.format(x) for x in card_bin_centres], frameon=False, loc='upper left', ncol=int(np.ceil(num_cards/3)))
# plt.ylim(0, 1.0)
plt.tight_layout()
plt.ioff()

if is_saving:
    plt.savefig(folder_name + 'Exp-DimPerCard_{}.pdf'.format(saving_label), format='pdf', dpi=333)

plt.show()