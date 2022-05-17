# Load data
import tools as tl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
import benchmark_func as bf
from operators import get_operator_aliases

sns.set(context="paper", font_scale=1, palette="husl", style="ticks",
        rc={'text.usetex': True, 'font.family': 'serif', 'font.size': 12,
            "xtick.major.top": False, "ytick.major.right": False})

is_saving = True
saving_format = 'png'
case = 1

chosen_categories = ['Differentiable', 'Unimodal']
case_label = 'DU'

# chosen_categories = ['Differentiable']
# case_label = 'D'

# chosen_categories = ['Unimodal']
# case_label = 'U'

#  ----------------------------------
# Read operators and find their alias
collection_file = 'default.txt'
with open('./collections/' + collection_file, 'r') as operators_file:
    heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]

num_search_operators = len(heuristic_space)
encoded_heuristic_space = np.arange(num_search_operators)

# --------------------------------
# Special adjustments for the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# Saving images flag
folder_name = 'data_files/transfer_learning/'
if is_saving:
    # Read (of create if so) a folder for storing images
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

# %% Main code section

datafile_names = ['unfolded_hhs_pop30', 'unfolded_hhs_pop50', 'unfolded_hhs_pop100']

def process_data(datafile_name):
    # Read the data file and assign the variables
    data_frame = tl.read_json(f'data_files/{datafile_name}.json')

    long_dimensions = data_frame['dimensions']
    long_problems = data_frame['problem']

    dimensions = sorted(list(set(long_dimensions)))
    problems = sorted(list(set(long_problems)))

    num_prob = len(problems)
    # num_oper = len(operators)
    num_dime = len(dimensions)

    # Call the problem categories
    problem_features = bf.list_functions(fts=chosen_categories)
    categories = sorted(set([problem_features[x]['Code'] for x in data_frame['problem']]), reverse=True)

    # Colour settings
    cmap = plt.get_cmap('tab20')
    colour_cat = [cmap(i)[:-1] for i in np.linspace(0, 1, len(categories))]
    colour_dim = [cmap(i)[:-1] for i in np.linspace(0, 1, num_dime)]


    # for basic metaheuristics
    current_performances = [x['performance'][-1] for x in data_frame['results']]

    # data_frame['results'][0].keys()
    # Out[23]: dict_keys(['step', 'performance', 'statistics', 'encoded_solution', 'hist_fitness'])

    # Create a data frame
    return pd.DataFrame({
        'Pop': [datafile_name.split('pop')[-1]] * len(long_dimensions),
        'Dim': [str(x) for x in data_frame['dimensions']],
        'Problem': data_frame['problem'],
        'Cat': [problem_features[x]['Code'] for x in data_frame['problem']],
        'uMH': [x['encoded_solution'][-1] for x in data_frame['results']]
    })
    # data_table.to_csv('data_files/unfolded_data.csv')


dataframes = list()
for datafile_name in datafile_names:
    dataframes.append(process_data(datafile_name))

# %%

total_df = pd.concat(dataframes, ignore_index=True)

# Get maximum number of search operators (i guess it was 100)
max_cardinality = max([len(x) for x in total_df['uMH']])

# Obtain the histogram per category
all_operators_including_empty = [-2.5, *np.arange(-2, num_search_operators) + 0.5]


def get_histogram(sequences):

    # print(len(sequences))
    # print(sequences)
    # Complete all sequences with -2 to get 100-cardinality metaheuristics
    mat_seq = np.array([np.array([*seq, *[-2] * (max_cardinality - len(seq))]) for seq in sequences],
                       dtype=object).T

    current_hist = list()

    for ii_step in range(max_cardinality):
        # Disregard the -2 and -1 operators (empty and initialiser)
        densities, _ = np.histogram(mat_seq[ii_step].tolist(), bins=all_operators_including_empty)
        temp_hist = densities[2:]
        if np.sum(temp_hist) > 0.0:
            current_hist.append(np.ndarray.tolist(temp_hist / np.sum(temp_hist)))
        else:
            current_hist.append(np.ndarray.tolist(np.ones(num_search_operators) / num_search_operators))

    return np.array(current_hist)


test = total_df.groupby(by=['Pop', 'Dim', 'Cat'], as_index=False)['uMH'].agg(get_histogram)
test['Dim'] = test['Dim'].apply(int)
test['Pop'] = test['Pop'].apply(int)
test = test.rename(columns={"uMH": "weights"})\
    .sort_values(by=['Pop', 'Dim', 'Cat'], ignore_index=True)

test.to_json("./data_files/translearn_dataset.json")

# %% ---------------------------
# % # # # # # # # # # # # # # # # # # # # #
# FIRST PLOT GROUP: HYPER-HEURISTIC LEVEL  #
# #  # # # # # # # # # # # # # # # # # # # #


for ind in range(len(test)):
    temp_dat = test.loc[ind]

    plt.figure(figsize=(3, 6))
    # plt.imshow(temp_dat['weights'].T, cmap="hot")
    sns.heatmap(temp_dat['weights'].T, robust=True)
    plt.xlabel('Step')
    plt.ylabel('Search Operator')
    plt.title(f"Pop: {temp_dat['Pop']}, Dim: {temp_dat['Dim']}, Cat: {temp_dat['Cat']}")

    if not is_saving:
        plt.savefig(folder_name + case_label + f"WM{ind}." + saving_format,
                    dpi=333, transparent=True)

    plt.show()
