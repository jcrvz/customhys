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

folder_name = 'images/'


# %% READ RAW DATA FILES
def read_data_files(main_folder_name='raw_data/'):
    # Define the basic data structure
    data = {'problem': list(), 'dimensions': list(), 'results': list()}

    # Get subfolder names: problem name & dimensions
    subfolder_names = os.listdir(main_folder_name)

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
            if (iteration == 0):
                initial_time = date_time
                absolute_time = 0
            else:
                absolute_time = (date_time - initial_time).total_seconds()

            # Read json file
            with open(temporal_full_path + '/' + iteration_file, 'r'
                      ) as json_file:
                temporal_data = json.load(json_file)

            # Store information in the correspoding variables
            iteration_data['iteration'].append(iteration)
            iteration_data['time'].append(absolute_time)
            for key in temporal_data.keys():
                iteration_data[key].append(temporal_data[key])

        # Store results in the main data frame
        data['results'].append(iteration_data)

    # Return only the data variable
    return data


# Read the data files
data_frame = read_data_files()

# %% PLOT FITNESS PER STEP

# Special adjustments
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Fitness evolution per replica
for problem_id in range(len(data_frame['problem'])):
    result = data_frame['results'][problem_id]

    plt.figure(problem_id, figsize=[4, 3], dpi=333)
    plt.ion()

    violin_parts = plt.violinplot([x['fitness'] for x in result['details']],
                                  result['iteration'],
                                  showmeans=True, showmedians=True,
                                  showextrema=False)

    violin_parts['cmeans'].set_edgecolor('#AC4C3D')  # Rojo
    violin_parts['cmeans'].set_linewidth(1.5)

    violin_parts['cmedians'].set_edgecolor('#285C6B')  # Azul
    violin_parts['cmedians'].set_linewidth(1.5)

    for vp in violin_parts['bodies']:
        vp.set_edgecolor('#154824')
        vp.set_facecolor('#4EB86E')
        vp.set_linewidth(1.0)
        vp.set_alpha(0.75)

    plt.ylabel(r'Fitness, $f(x)$')
    plt.xlabel(r'Step')
    plt.title(r'' + '{} {}D'.format(data_frame['problem'][problem_id],
                                    data_frame['dimensions'][problem_id]))
    plt.ioff()
    plt.legend([Line2D([0], [0], color='#AC4C3D', lw=3),
                Line2D([0], [0], color='#285C6B', lw=3)],
               ['Mean', 'Median'], frameon=False)

    file_name = '{}-{}D'.format(data_frame['problem'][problem_id],
                                data_frame['dimensions'][problem_id])
    plt.savefig(folder_name + 'vp' + file_name + '.eps',
                format='eps', dpi=1000)
    plt.show()

# %% PLOT FITNESS PER REPLICA (DETAILS)
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
    plt.title(r'' + '{} {}D'.format(
        data_frame['problem'][problem_id],
        data_frame['dimensions'][problem_id]))

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
    plt.savefig(folder_name + 'it' + file_name + '.eps',
                format='eps', dpi=1000)
    plt.show()
