# -*- coding: utf-8 -*-


import os
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler
import mpl_toolkits.mplot3d
import numpy as np
from tools import *


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

problems = list(set(data_frame['problem']))
dimensions = list(set(data_frame['dimensions']))
operators = list(set(data_frame['results'][0]['operator_id']))

# Show the variable tree
printmsk(data_frame)

# %% PLOT FITNESS PER CARD/DIMENSION
is_saving = False

# Special adjustments
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# Initialise variables
empty_dict = dict(Min=list(), Med=list(), Avg=list(), Std=list(),
                  Max=list(), IQR=list(), MAD=list())
stats = empty_dict.copy()

# Plot a figure per dimension
for dimension in dimensions:
    # Find indices corresponding to such a dimension
    dim_indices = listfind(data_frame['dimensions'], dimension)

    # Get temporal stat dictionary
    temp_stats = empty_dict.copy()

    # Get fitness statistical values and stored them in 'arrays'
    for dim_index in dim_indices:
        for op_index in range(len(operators)):
            current_stats = data_frame['results'][dim_index][
                    'statistics'][op_index]
            for key in temp_stats.keys():
                temp_stats[key].append(current_stats[key])

    # Store temporal stat dictionary into the stats dictionary
    for key, val in temp_stats.items():
        stats[key].append([*val])

    matrix_z = np.array(stats)

    fig = plt.figure(figsize=[3, 4], dpi=333)
    plt.ion()

    ls = LightSource(azdeg=90, altdeg=45)
    rgb = ls.shade(matrix_z, plt.cm.jet)

    ax = fig.gca(projection='3d')
    ax.plot_surface(matrix_x, matrix_y, matrix_z, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, facecolors=rgb)

# %%

# Fitness evolution per replica
for problem_str in problems:
    plt.figure(figsize=[3, 4], dpi=333)
    plt.ion()

    y_data = []
    for dimension in dimensions:
        for problem_id in range(len(data_frame['problem'])):
            if ((data_frame['problem'][problem_id] == problem_str) and
                    (data_frame['dimensions'][problem_id] == dimension)):
                result = data_frame['results'][problem_id]

                y_data.append(np.log10(np.array(
                    result['details'][-1]['fitness']) + 1.0))

    violin_parts = plt.violinplot(y_data, range(len(y_data)),
                                  showmeans=True, showmedians=True, showextrema=False)

    violin_parts['cmeans'].set_edgecolor('#AC4C3D')  # Rojo
    violin_parts['cmeans'].set_linewidth(1.5)

    violin_parts['cmedians'].set_edgecolor('#285C6B')  # Azul
    violin_parts['cmedians'].set_linewidth(1.5)

    for vp in violin_parts['bodies']:
        vp.set_edgecolor('#3B1255')
        vp.set_facecolor('#9675AB')
        vp.set_linewidth(1.0)
        vp.set_alpha(1.0)

    plt.xticks(range(len(y_data)), dimensions)
    plt.ylim(ylims[problem_str])
    # plt.yscale('log')
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.ylabel(r'Fitness, $\log(f(x) + 1)$')
    plt.xlabel(r'Dimensions')

    plt.title(r'' + 'Cardinality: ' + cardinality, loc='center')

    plt.ioff()
    if cardinality == '1':
        plt.legend([Line2D([0], [0], color='#AC4C3D', lw=3),
                    Line2D([0], [0], color='#285C6B', lw=3)],
                   ['Mean', 'Median'], frameon=False,
                   loc="upper left", borderaxespad=0, ncol=1)
    # bbox_to_anchor=(0, 0.9, 1, 0.2),
    file_name = '{}'.format(problem_str)
    plt.tight_layout()
    if is_saving:
        plt.savefig(folder_name + 'vp' + file_name + '.eps',
                    format='eps', dpi=1000)
        print(file_name + ' Saved!')
    plt.show()

# %% PLOT FITNESS PER STEP
is_saving = False

# Special adjustments
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Fitness evolution per replica
for problem_id in range(len(data_frame['problem'])):
    result = data_frame['results'][problem_id]

    plt.figure(problem_id, figsize=[4, 3], dpi=333)
    plt.ion()

    # np.log10(x['fitness'] + 1.0)
    violin_parts = plt.violinplot([
        np.log10(np.array(x['fitness']) + 1.0)
        for x in result['details']], result['iteration'],
        showmeans=True, showmedians=True, showextrema=False)

    violin_parts['cmeans'].set_edgecolor('#AC4C3D')  # Rojo
    violin_parts['cmeans'].set_linewidth(1.5)

    violin_parts['cmedians'].set_edgecolor('#285C6B')  # Azul
    violin_parts['cmedians'].set_linewidth(1.5)

    for vp in violin_parts['bodies']:
        vp.set_edgecolor('#154824')
        vp.set_facecolor('#4EB86E')
        vp.set_linewidth(1.0)
        vp.set_alpha(0.75)

    plt.xticks([0, 1, 2], [0, 1, 2])
    # plt.ylim(bottom=0.9)
    # plt.yscale('log')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.ylabel(r'Fitness, $\log(f(x) + 1)$')
    plt.xlabel(r'Step')
    # plt.title(r'' + '{} {}D'.format(data_frame['problem'][problem_id],
    #                                 data_frame['dimensions'][problem_id]))
    plt.ioff()
    plt.legend([Line2D([0], [0], color='#AC4C3D', lw=3),
                Line2D([0], [0], color='#285C6B', lw=3)],
               ['Mean', 'Median'], frameon=False)

    file_name = '{}-{}D'.format(data_frame['problem'][problem_id],
                                data_frame['dimensions'][problem_id])
    plt.tight_layout()
    if is_saving:
        plt.savefig(folder_name + 'vp' + file_name + '.eps',
                    format='eps', dpi=1000)
    plt.show()

# %% PLOT FITNESS PER REPLICA (DETAILS)
is_saving = False

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rc('figure', figsize=[4, 3], dpi=333)

for problem_id in range(len(data_frame['problem'])):
    result = data_frame['results'][problem_id]

    num_solutions = len(result['iteration'])

    cmap = plt.cm.jet
    colours = cmap(np.linspace(0.1, 0.9, num_solutions))
    rcParams['axes.prop_cycle'] = cycler(color=colours)

    plt.figure(problem_id, figsize=[4, 3], dpi=333)

    plt.xlabel(r'Iteration')
    # plt.title(r'' + '{} {}D'.format(
    #     data_frame['problem'][problem_id],
    #     data_frame['dimensions'][problem_id]))

    plt.ion()
    plt.ylabel(r'Fitness, $f(x) + 1$')

    for solution_id in range(num_solutions):
        for xx in result['details'][solution_id]['historical']:
            plt.semilogy([y + 1 for y in xx['fitness']], lw=0.5,
                         color=colours[solution_id])
    plt.ioff()
    plt.legend([Line2D([0], [0], color=colour, lw=3) for colour in colours],
               [r'Step ' + f'{sol}' for sol in result['iteration']],
               frameon=False)
    # plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

    file_name = '{}-{}D'.format(data_frame['problem'][problem_id],
                                data_frame['dimensions'][problem_id])
    plt.tight_layout()
    if is_saving:
        plt.savefig(folder_name + 'it' + file_name + '.eps',
                    format='eps', dpi=1000)
    plt.show()
