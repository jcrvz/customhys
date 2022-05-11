import igraph as ig  # pip install python-igraph
import pandas as pd
import os
import numpy as np


# %% This part correspond to the create.R port

input_folder = './data_files/toSTN/Sphere'
num_runs = 30
minimum_val = 0

output_folder = input_folder + '-stn'

# just for testing
instance = "MH1_Sphere_5D.txt"


def check_inputs(input_folder, num_runs, minimum_val):
    if not os.path.isdir(input_folder):
        raise STN_Error("Input folder does not exist!")

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    if not (isinstance(num_runs, int) and num_runs > 1):
        raise STN_Error("Invalid number of runs")

    if not minimum_val:
        raise STN_Error("Invalid minimum value")


# def create(instance):
#     if instance:
file_name = input_folder + '/' + instance

# Read data from file
df = pd.read_table(file_name, header=0, delimiter='\t',
                   names=['Run', 'Fitness1', 'Solution1', 'Fitness2', 'Solution2'],
                   dtype={'Run': np.int32, 'Fitness1': np.float64, 'Solution1': str,
                          'Fitness2': np.float64, 'Solution2': str}
                   )

# Trim data up to num_runs
df = df[df['Run'] <= num_runs]

# Initialise some variables
nodes = list()
edges = list()
start_ids = list()
end_ids = list()




    # else:
    #     raise STN_Error("There is not instance")





class STN_Error(Exception):
    """
    Simple STN Error to manage exceptions.
    """
    pass
