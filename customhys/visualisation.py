# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:37:54 2020

@author: L03130342
"""

import os
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler
import numpy as np


# READ RAW DATA FILES
def read_data_files(main_folder_name='raw_data/'):
    # Define the basic data structure
    data = {'problem': list(), 'dimensions': list(), 'results': list()}

    # Get subfolder names: problem name & dimensions
    subfolder_names = sorted(os.listdir(main_folder_name),
                             key=lambda x: int(x.split('-')[1].strip('D')))

    for subfolder in subfolder_names:
        # Extract the problem name and the number of dimensions
        problem_name, dimensions, date_str = subfolder.split('-')

        # Store information about this subfolder
        data['problem'].append(problem_name)
        data['dimensions'].append(int(dimensions[:-1]))

        # Read all the iterations files contained in this subfolder
        temporal_full_path = os.path.join(main_folder_name, subfolder)
        iteration_file_names = os.listdir(temporal_full_path)

        # Sort the list of files based on their iterations
        iteration_file_names = sorted(iteration_file_names,
                                      key=lambda x: int(x.split('-')[0]))

        # Initialise iteration data with same field as files
        iteration_data = {'iteration': list(), 'time': list(),
                          'encoded_solution': list(), 'solution': list(),
                          'performance': list(), 'details': list()}

        # Walk on subfolder's files
        for iteration_file in tqdm(iteration_file_names,
                                   desc='{} {}'.format(
                                       problem_name, dimensions)):
            # Extract the iteration number and time
            iteration_str, time_str = iteration_file.split('-')
            iteration = int(iteration_str)

            # Determine the absolute times (in seconds)
            date_time = datetime.strptime(time_str + date_str,
                                          '%H_%M_%S.json%m_%d_%Y')
            if iteration == 0:
                initial_time = date_time
                absolute_time = 0
            else:
                absolute_time = (date_time - initial_time).total_seconds()

            # Read json file
            with open(temporal_full_path + '/' + iteration_file, 'r'
                      ) as json_file:
                temporal_data = json.load(json_file)

            # Store information in the corresponding variables
            iteration_data['iteration'].append(iteration)
            iteration_data['time'].append(absolute_time)
            for key in temporal_data.keys():
                iteration_data[key].append(temporal_data[key])

        # Store results in the main data frame
        data['results'].append(iteration_data)

    # Return only the data variable
    return data


# Read the data files
cardinality = '3'
data_frame = read_data_files('raw_data-card' + cardinality)
folder_name = 'images-card' + cardinality + '/'

problems = list(set(data_frame['problem']))
dimensions = list(set(data_frame['dimensions']))

# %% PRINT THE BEST METAHEURISTIC
for problem_str in problems:
    for dimension in dimensions:
        for problem_id in range(len(data_frame['problem'])):
            if ((data_frame['problem'][problem_id] == problem_str) and
                    (data_frame['dimensions'][problem_id] == dimension)):
                # Get the corresponding data
                result = data_frame['results'][problem_id]

                # Obtain stats
                stats = result['details'][-1]['statistics']

                # Read the solution
                solution = result['solution'][-1]
                solution_str = ['{} with {}'.format(
                    sol[0], str(sol[1])) for sol in solution]

                print(" & ".join([
                    problem_str, str(dimension),
                    *['{:.4g}'.format(stats[st]) for st in [
                        'Avg', 'Std', 'Med', 'IQR', 'Min', 'Max']],
                    *solution_str]), end='')
                print('\\\\')

# %% PLOT FITNESS PER CARD/DIMENSION
is_saving = True

# Special adjustments
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# Set of problems and dimensions stored
ylims = {'Sphere': (-0.1, 4.1), 'Griewank': (-0.05, 1.05),
         'Ackley': (-0.05, 1.25), 'Rosenbrock': (-0.1, 2.6)}

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
