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
import pandas as pd
import seaborn as sns

# %% READ RAW DATA FILES

def read_data_files(main_folder_name='raw_data/'):
    # Define the basic data structure
    empty_data_structure = {'iteration': list(), 'time': list(),
                            'encoded_solution': list(), 'solution': list(),
                            'performance': list(), 'details': list()}
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
                
                # Initialise iteration data with same field as files
                iteration_data = empty_data_structure
            else:
                absolute_time = (date_time - initial_time).total_seconds()
            
            # Read json file
            with open(temporal_full_path + '/' + iteration_file,
                      'r') as json_file:
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

# %% PLOT SOME FIGURES

# Special adjustments
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Fitness evolution per replica
problem = 'Griewank'
dimension = 2

for problem_id in range(len(data_frame['problem'])):
    result = data_frame['results'][problem_id]
    
    # plt.plot(result['iteration'], result['performance'])
    plt.figure(problem_id, figsize=[4, 3], dpi=333)
    plt.ion()
    # plt.plot(result['iteration'], [x['statistics']['Avg'] for x
    #                                in result['details']],
    #          'b', linewidth=1.5, label='Average')
    # plt.plot(result['iteration'], [x['statistics']['Med'] for x
    #                                in result['details']],
    #          'r', linewidth=1.5, label='Median')
    # plt.plot(result['iteration'], [x['statistics']['Min'] for x
    #                                in result['details']],
    #          'c', linewidth=1.5, label='Min.')    
    # plt.plot(result['iteration'], [x['statistics']['Max'] for x
    #                                in result['details']],
    #          'm', linewidth=1.5, label='Max.')
    plt.violinplot([x['fitness'] for x in result['details']],
                   showmeans=True, showmedians=True, showextrema=False)
    # plt.ylim((-1.0, -0.5))
    plt.ylabel(r'Metric')
    plt.xlabel(r'Step')
    plt.ioff()
    # plt.legend(frameon=False)
    plt.show()
