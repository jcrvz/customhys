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
import seaborn as sns
import pandas as pd
import benchmark_func as bf
from scipy import stats
sns.set(context="paper", font_scale=1, palette="colorblind", style="ticks",
        rc={'text.usetex':True, 'font.family':'serif', 'font.size':12})

# Read benchmark functions and their features
problem_features = bf.list_functions()
problem_names = bf.for_all('func_name')

# DEL: Read benchmark functions, their features and categorise them
# problem_features_without_code = bf.list_functions()

# %% CASES

# Set the test case
test_case = 1

# Saving images flag
is_saving = False

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
basic_mhs_data = jt.read_json('data_files/basic-metaheuristics-data_v2.json')
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
        success_rate.append(np.sum(performance_comparison <= 0.0) / len(performance_comparison))

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
    data_per_dimension.append(current_data_per_dimension)  # sort_index()

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
    dummy_df = data_per_dimension[dim_ind].sort_index()

    z1 = dummy_df['success-Rate']

    ax.plot3D(x1, y1, z1)

plt.ioff()

plt.yticks(np.arange(len(y1)), dimensions)
plt.ylim(0, len(dimensions)-1)
plt.xlim(1, x1[-1] + 1 )
ax.set_zlim(-0.01, 1.01)
ax.set_xticks(np.round(np.linspace(1, 107, 7)))
ax.set_ylabel(r'Dimension')
ax.set_xlabel(r'Problem Id.')
ax.set_zlabel(r'Success Rate')
plt.tight_layout()

if is_saving:
    plt.savefig(folder_name + 'Exp-SuccessRatePerDimFunc_{}.eps'.format(saving_label), format='eps',
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
plt.ylim((-0.1, 1.1))
# plt.yscale('log')
plt.xlim(0, len(dimensions)-1)
plt.xticks(np.arange(len(dimensions)), dimensions)
plt.tight_layout()

if is_saving:
    plt.savefig(folder_name + 'Exp-SuccessRatePerDimCat_{}.pdf'.format(saving_label), format='pdf', dpi=333)

plt.show()

# %% P-Value FOR ALL DIMENSIONS TOTAL

# fig = plt.figure(f)

fig, axs = plt.subplots(len(dimensions), sharex=True, sharey=True, figsize=(4.5, 7), dpi=125, facecolor='w')

cmap = plt.get_cmap('tab10')
colors = [cmap(i)[:-1] for i in np.linspace(0, 1, len(dimensions))]

categories_r = categories[::-1]
frames = [pd.DataFrame(x.groupby('Category')["p-Value"].agg('mean')) for x in data_per_dimension]
result = pd.concat(frames, axis=1)
result.columns = ['{}D'.format(dim) for dim in dimensions]
print(result.iloc[::-1].to_latex(index=True, float_format="%.3g"))

avg_p_value = [np.nan_to_num(x["p-Value"].values) for x in data_per_dimension]
pVal_stats = stats.describe(avg_p_value, axis=1)
print(pd.DataFrame({"mean": pVal_stats.mean, "std": np.sqrt(pVal_stats.variance)},
                   index=dimensions).to_latex())

for id_dim in range(len(dimensions)):
    pValues_per_dim = data_per_dimension[id_dim].groupby('Category')["p-Value"].apply(list)

    y_data = [np.nan_to_num(list(x)) for x in pValues_per_dim.values][::-1]
    x_data = np.arange(len(categories_r))

    violin_parts = axs[id_dim].violinplot(np.array(y_data), x_data,
            showmeans=True, showmedians=True, showextrema=False)

    # axs[id_dim].set_ylim(0, 0.05)
    # axs[id_dim].set_yscale('log')
    axs[id_dim].set_title(r'{}D'.format(dimensions[id_dim]), fontsize=12)

    violin_parts['cmeans'].set_edgecolor('#AC4C3D')  # Rojo
    violin_parts['cmeans'].set_linewidth(1.5)

    violin_parts['cmedians'].set_edgecolor('#285C6B')  # Azul
    violin_parts['cmedians'].set_linewidth(1.5)

    for vp in violin_parts['bodies']:
        vp.set_facecolor(colors[id_dim])
        vp.set_linewidth(1.0)
        vp.set_alpha(0.5)


axs[3].set_ylabel(r'$p$-Value')
plt.xlabel(r'Categories')
plt.ioff()
# plt.yscale('log')
axs[0].legend([Line2D([0], [0], color='#AC4C3D', lw=3),
            Line2D([0], [0], color='#285C6B', lw=3)],
           ['Mean', 'Median'], frameon=False, loc='center right', fontsize=12)
plt.xlim(-0.5, x_data[-1]+0.5)
plt.xticks(x_data, labels=categories_r)
plt.tight_layout()

if is_saving:
    plt.savefig(folder_name + 'Exp-PValue_{}.pdf'.format(saving_label), format='pdf', dpi=333)

plt.show()

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

#  %%
#
# # %% Print violin plots
#
# # Printing section
# fig, axes = plt.subplots(figsize=(0.08*205, 0.4*8), dpi=333)
#
# violin_parts = plt.violinplot(np.array(temporal_array), stats_without_category.columns.to_list(),
#                               showmeans=True, showmedians=True, showextrema=False)
#
# violin_parts['cmeans'].set_edgecolor('#AC4C3D')  # Rojo
# violin_parts['cmeans'].set_linewidth(1.5)
#
# violin_parts['cmedians'].set_edgecolor('#285C6B')  # Azul
# violin_parts['cmedians'].set_linewidth(1.5)
#
# for vp in violin_parts['bodies']:
#     vp.set_edgecolor('#154824')
#     vp.set_facecolor('#4EB86E')
#     vp.set_linewidth(1.0)
#     vp.set_alpha(0.75)
#
# axes.set_xticks(stats_without_category.columns.to_list())
# plt.ylabel(r'Rank')
# plt.xlabel(r'Simple Metaheuristic')
#
# plt.legend([Line2D([0], [0], color='#AC4C3D', lw=3),
#             Line2D([0], [0], color='#285C6B', lw=3)],
#            ['Mean', 'Median'], frameon=False)
#
# # plt.title("Dim: {}".format(dimension))
# plt.tight_layout()

# %%
dfs = [x['Performance'].sort_index() for x in data_per_dimension]
result_v2 = pd.concat(dfs, axis=1)
result_v2.columns = ['{}D'.format(dim) for dim in dimensions]
for col in result_v2.columns:
    result_v2[col] = result_v2[col].map('{:.4g}'.format)
# print(result_v2.to_latex(index=True, float_format="{:0.3f}".format ))
# data_per_dimension.to_csv('data_files/first_pd{}D.csv'.format(dimension), index=False)
with open('data_files/first_test{}.tex'.format(test_case), 'w') as tf:
    tf.write(result_v2.to_latex())

# %% Granular analysis

# Selected problems
selected_problems_names = ['Sphere', 'Rastrigin', 'Schwefel', 'Griewank', 'Step',
                     'Stochastic', 'TypeI', 'SchafferN3']
custom_ylims = {'Sphere':(-1, 100), 'Rastrigin':(-1, 110), 'Schwefel':(-0.1, 22), 'Griewank':(-0.01, 6),
                'Step':(-1, 175), 'Stochastic':(-0.1, 9), 'SchafferN3':(-0.1, 20), 'NeedleEye':(-100, 4000)}
cmap = plt.get_cmap('tab10')
colors = [cmap(i)[:-1] for i in np.linspace(0, 1, len(dimensions))]
boxprops = []
whiskerprops = []
medianprops = []
meanprops = []
capprops = []
for col in colors:
    boxprops.append(dict(linestyle='-', linewidth=1, edgecolor=col, facecolor=col))
    whiskerprops.append(dict(linestyle='-', linewidth=1, color=col))
    medianprops.append(dict(linestyle='-', linewidth=1, color='black'))
    meanprops.append(dict(linestyle='--', linewidth=1, color='blue'))
    capprops.append(dict(linestyle='-', linewidth=1, color=col))

# For each selected problem get the ids
# for problem in selected_problems_names:
#     selected_problem_ids = [i for i,x in enumerate(new_mhs_data['problem']) if x == problem]
#
#     fig, ax = plt.subplots(1, figsize=[3, 4], dpi=333)
#     plt.ion()
#
#     # For each dimension
#     for dim_id in range(len(dimensions)):
#         datum = new_mhs_data['results'][selected_problem_ids[dim_id]]['statistics'][-1]
#
#         pre_boxplot = []
#         # for datum in data_set:
#         item = {}
#
#         # item["label"] = '{}'.format(dimensions[dim_id])  # not required
#         item["mean"] = datum['Avg']  # not required
#         item["med"] = datum['Med']
#         item["q1"] = datum['Med'] - datum['IQR']/2
#         item["q3"] = datum['Med'] + datum['IQR']/2
#         item["whislo"] = datum['Min']  # required
#         item["whishi"] = datum['Max']  # required
#         item["fliers"] = []  # required if showfliers=True
#
#         pre_boxplot.append(item)
#
#         ax.bxp(pre_boxplot, positions=[dim_id], patch_artist=True, widths=0.5, meanprops=meanprops[dim_id],
#                boxprops=boxprops[dim_id], capprops=capprops[dim_id], showmeans=True, meanline=True,
#                whiskerprops=whiskerprops[dim_id], medianprops=medianprops[dim_id])
#
#         ax.plot(new_mhs_data['results'][selected_problem_ids[dim_id]]['hist_fitness'])
#
#     ax.set_title(problem)
#     ax.set_xlabel(r'Dimensions')
#     ax.set_ylim(custom_ylims[problem])
#     ax.set_ylabel(r'Finess values')
#     ax.set_xticklabels(dimensions)
#     plt.ioff()
#     plt.show()


# %% ---
# For each selected problem get the ids
for problem in selected_problems_names:
    selected_problem_ids = [i for i, x in enumerate(new_mhs_data['problem']) if x == problem]

    fig, axs = plt.subplots(1, figsize=[4, 2.5], dpi=333)
    plt.ion()

    # For each dimension
    for dim_id in range(len(dimensions)):
        matrix = np.array(new_mhs_data['results'][selected_problem_ids[dim_id]]['hist_fitness']).T

        axs.plot(matrix, color=colors[dim_id], alpha=0.05, linewidth=2)
        axs.plot(np.mean(matrix, 1), color=colors[dim_id])

    # axs.set_title(problem)
    axs.set_xlabel(r'Iteration')
    # ax.set_ylim(custom_ylims[problem])
    axs.set_ylabel(r'Fitness values')
    # ax.set_xticklabels(dimensions)
    plt.ioff()

    lines_for_legend = []
    dimensions_for_legend = []
    if problem == 'Sphere':
        for dim_id in range(len(dimensions)):
            lines_for_legend.append(Line2D([0], [0], color=colors[dim_id], lw=3))
            dimensions_for_legend.append('{}D'.format(dimensions[dim_id]))
        axs.legend(lines_for_legend, dimensions_for_legend, ncol=2,
                   frameon=False, loc='upper right', fontsize=12)
    plt.tight_layout()

    if is_saving:
        plt.savefig(folder_name + 'Sample_{}-Exp{}.pdf'.format(problem, test_case),
                    format='pdf', dpi=333)

    plt.show()

# new_mhs_data['results'][selected_problem_ids[dim_id]]['hist_fitness']

# %%
so_labels = ['{}'.format(cardi+1) for cardi in range(card_upto)]


operator_families = [operator.split(',')[0].capitalize() for operator in search_operators]
os_family = sorted(list(set(operator_families)), reverse=True)

fig, axs = plt.subplots(1, len(dimensions), figsize=[10, 3], dpi=333, sharey=True)

# libraries
for dim_id in range(len(dimensions)):
    df0 = data_per_dimension[dim_id].copy()
    df0['Metaheuristic'] = df0['Metaheuristic'].apply(
        lambda x: [os_family.index(operator_families[int(y)]) for y in x])
    df1 = pd.DataFrame(df0["Metaheuristic"].to_list(), columns=so_labels)

    df2 = pd.concat([data_per_dimension[dim_id]['Category'], df1], axis=1)
    df2.Category = df2.Category.astype('category')

    # Make the plot
    pd.plotting.parallel_coordinates(df2, 'Category', colormap=plt.get_cmap("tab10"),
                                     sort_labels=True, alpha=0.3, ax=axs[dim_id])

    axs[dim_id].set_title(r'{}D'.format(dimensions[dim_id]))

    if dim_id == 0:
        handles, labels = axs[dim_id].get_legend_handles_labels()

        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
        axs[dim_id].legend(handles, labels, loc="upper center", ncol=len(dimensions) + 1,
                           bbox_to_anchor=(4.5, 0.5))
    else:
        axs[dim_id].get_legend().remove()

axs[0].set_yticks(np.arange(len(os_family)))
axs[0].set_yticklabels(os_family)
axs[3].set_xlabel(r'Heuristic')

if is_saving:
    plt.savefig(folder_name + 'ParCoor-Exp{}.svg'.format(test_case), format='svg', dpi=333)

plt.show()

# %%
def real_formatting(value):
    if value == 0.0:
        value_order = 1
    else:
        value_order = np.floor(np.log10(np.abs(value)))
    if -2 <= value_order <= 2:
        return '%.2f' % value
    else:
        return '%.1fe%d' % (value * (10 ** -value_order), value_order)

def get_subframework(fields, with_cat=False, alias=None):
    if not isinstance(fields, list):
        fields = [fields]
    def mask(x):
        if alias is None:
            return  x[:4]
        else:
            return alias

    cat = ['Category'] if with_cat else []
    frs = [data_per_dimension[0][cat + fields].sort_index()] + \
          [x[fields].sort_index() for x in data_per_dimension[1:]]
    rsl = pd.concat(frs, axis=1)
    # rsl.columns = sum( (['MH{}D'.format(x), 'Perf{}D'.format(x)] for x in dimensions), [] )
    rsl.columns = cat + ['{}{}D'.format(mask(y), x) for x in dimensions for y in fields]
    return rsl

rsl = get_subframework(['Metaheuristic', 'Performance'], with_cat=False, alias='')

for col in ['Perf{}D'.format(x) for x in dimensions]:
    rsl[col] = rsl[col].apply(real_formatting)
for col in ['Meta{}D'.format(x) for x in dimensions]:
    rsl[col] = rsl[col].apply(lambda y: ', '.join([str(x) for x in y]))

print(rsl.to_latex())

# %%

rsl1 = get_subframework(['p-Value'], with_cat=True)

fig = plt.figure(figsize=[4, 3], dpi=333)

colours1 = plt.cm.tab10(np.linspace(0, 1, len(dimensions)))

# for dim, c in zip(dimensions[1:], colours1):
#     if dim == 2:
#         sns.stripplot(rsl1.Category, rsl1['p-Va2D'], jitter=0.01, size=3, color=c,
#                            dodge=False, label='2D')
#         # ax = rsl1.plot.scatter(x='Category', y='p-Va2D', color=c, label='2D')
#     else:
#         # rsl1.plot.scatter(x='Category', y='p-Va{}D'.format(dim), color=np.array([c]),
#         #                   label=r'{}D'.format(dim), ax=ax)
#         sns.stripplot(rsl1.Category, rsl1['p-Va{}D'.format(dim)], jitter=0.1, size=3, color=c,
#                       dodge=False, label=r'{}D'.format(dim))

dff = pd.melt(rsl1, var_name='Dim', value_vars=['p-Va{}D'.format(x) for x in dimensions],
              id_vars=['Category'], value_name='p-Value')
sns.stripplot(data = dff, x='Category', y = 'p-Value', hue = 'Dim', size=3,
              jitter = 0.1, dodge = True, alpha = 0.7,  palette = colours1)
plt.show()
