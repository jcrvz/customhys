# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:37:54 2020

@author: L03130342
"""

import os
from datetime import datetime

# DEFINE THE BASIC DATA STRUCTURE
def get_empty_structure():
    return {'iterations': list(), 'time': list(), 'econded_solution': list(),
            'solution': list(), 'performances': list(), 'details': list()}
data = {'problem': list(), 'dimensions': list(), 'results': list()}

# %% READ RAW DATA FILES

# Specify basic information
main_folder_name = 'raw_data/'

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
    
    # Walk on subfolder's files
    for iteration_file in iteration_file_names:
        # Extract the iteration number and time
        iteration_str, time_str = iteration_file.split('-')
        
        date_time = datetime.strptime(time_str + ' ' + date_str,
                                      '%H_%M_%S.json %d_%m_%Y')
        print(date_time)

