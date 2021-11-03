"""
This module contains tools for processing and dealing with some data liaised to this framework.

Created on Sat Feb 22, 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""
import random
import os
import json
from subprocess import call
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy.stats as st


def printmsk(var, level=1, name=None):
    """
    Print the meta-skeleton of a variable with nested variables, all with different types.

    Example:

    >>> variable = {"par0": [1, 2, 3, 4, 5, 6],
            "par1": [1, 'val1', 1.23],
            "par2" : -4.5,
            "par3": "val2",
            "par4": [7.8, [-9.10, -11.12, 13.14, -15.16]],
            "par5": {"subpar1": 7,
                     "subpar2": (8, 9, [10, 11])}}

    >>> printmsk(variable)
    |-- {dict: 6}
    |  |-- par0 = {list: 6}
    |  |  |-- 0 = {int}
    :  :  :
    |  |-- par1 = {list: 3}
    |  |  |-- 0 = {int}
    |  |  |-- 1 = {str}
    |  |  |-- 2 = {float}
    |  |-- par2 = {float}
    |  |-- par3 = {str}
    |  |-- par4 = {list: 2}
    |  |  |-- 0 = {float}
    |  |  |-- 1 = {list: 4}
    |  |  |  |-- 0 = {float}
    :  :  :  :
    |  |-- par5 = {dict: 2}
    |  |  |-- subpar1 = {int}
    |  |  |-- subpar2 = {tuple: 3}
    |  |  |  |-- 0 = {int}
    |  |  |  |-- 1 = {int}
    |  |  |  |-- 2 = {list: 2}
    |  |  |  |  |-- 0 = {int}
    :  :  :  :  :

    :param any var:
        Variable to inspect.
    :param int level: Optional.
        Level of the variable to inspect. Default: 1.
    :param name: Optional.
        Name of the variable to inspect. It is just for decorative purposes. The default is None.
    :return: None.
    """
    # Parent inspection
    parent_type = var.__class__.__name__
    var_name = "" if name is None else name + " = "
    print('|  ' * (level - 1) + '|-- ' + var_name + "{", end="")

    # Check if it has __len__ but is not str or ndarray
    if hasattr(var, '__len__') and not (parent_type in ['str', 'ndarray']):
        print('{}: {}'.format(parent_type, len(var)) + '}')

        # If is it a dictionary
        if parent_type == 'dict':
            for key, val in var.items():
                printmsk(val, level + 1, str(key))
        elif parent_type in ['list', 'tuple']:
            # Get a sample: first 10 elements (if the list is too long)
            if len(var) > 10:
                var = var[:10]

            # If all the elements has same type, then show an example
            if len(set([val.__class__.__name__ for val in var])) == 1:
                printmsk(var[0], level + 1, '0')
                print(':  ' * (level + 1))
            else:
                for iid in range(len(var)):
                    printmsk(var[iid], level + 1, str(iid))
    else:
        if parent_type == 'ndarray':
            dimensions = ' x '.join([str(x) for x in var.shape])
            print('{}: {}'.format(parent_type, dimensions) + '}')
        else:
            print('{}'.format(parent_type) + '}')


def listfind(values, val):
    """
    Return all indices of a list corresponding to a value.

    :param list values:
        List to analyse.
    :param any val:
        Element to find in the list.
    :return: list
    """
    return [i for i in range(0, len(values)) if values[i] == val]


def revise_results(main_folder='data_files/raw/'):
    """
    Revise a folder with subfolders and check if there are subfolder repeated, in name, then merge. The repeated
    folders are renamed by adding the prefix '.to_delete-', but before merge their data into a unique folder.

    :param str main_folder: Optional.
        Path to analyse. The default is 'data_files/raw/'.
    :return: None
    """
    raw_folders = [element for element in os.listdir(main_folder) if not element.startswith('.')]
    folders_with_date = sorted(raw_folders, key=lambda x: x.split('D-')[0])
    folders_without_date = [x.split('D-')[0] for x in folders_with_date]

    # Look for repeated folder names without date
    for folder_name in list(set(folders_without_date)):
        indices = listfind(folders_without_date, folder_name)
        if len(indices) > 1:
            # Merge this folders into the first occurrence
            destination_folder = main_folder + folders_with_date[indices[0]]

            for index in indices[1:]:
                # Copy all content to the first folder
                call(['cp', '-a', main_folder + folders_with_date[index] + '/*', destination_folder])

                # Rename the copied folder with prefix '.to_delete-'
                call(['mv', main_folder + folders_with_date[index],
                      main_folder + '.to_delete-' + folders_with_date[index]])
                print("Merged '{}' into '{}'!".format(folders_with_date[index], folders_with_date[indices[0]]))


def read_subfolders(foldername):
    """
    Return a list of all subfolders contained in a folder, ignoring all those starting with '.' (hidden ones).

    :param str foldername:
        Name of the main folder.
    :return: list.
    """
    return [element for element in os.listdir(foldername) if not element.startswith('.')]


def preprocess_files(main_folder='data_files/raw/', kind='brute_force', only_laststep=True,
                     output_name='processed_data', experiment=''):
    """
    Return data from results saved in the main folder. This method save the summary file in json format. Take in account
    that ``output_name = 'brute_force'`` has a special behaviour due to each json file stored in sub-folders correspond
    to a specific operator. Otherwise, these files use to correspond to a candidate solution (i.e., a metaheuristic)
    from the hyper-heuristic process.

    :param str main_folder: Optional.
        Location of the main folder. The default is 'data_files/raw/'.
    :param str kind:
        Type of procedure run to obtain the data files. They can be 'brute_force', 'basic_metaheuristic', and any other,
        which means metaheuristics without fixed search operators. The default is 'brute_force'.
    :param bool only_laststep: Optional.
        Flag for only save the last step of all fitness values from the historical data. It is useful for large amount
          of experiments. It only works when ``kind'' is neither 'brute_force' or 'basic_metaheuristic'. The default is
          True.
    :param str output_name:
        Name of the resulting file. The default is 'processed_data'.

    :return: dict.
    """
    # TODO: Revise this method to enhance its performance.
    # Get folders and exclude hidden ones
    raw_folders = filter(lambda name: len(name.split('-')) >= 2, read_subfolders(main_folder))

    # Sort subfolder names by problem name & dimensions
    subfolder_names_raw = sorted(raw_folders, key=lambda x: int(x.split('-')[1].strip('D')))

    if len(experiment) > 0:
        subfolder_names = filter(lambda name: len(name.split('-')) >= 3 and name.split('-')[2] == experiment, subfolder_names_raw)
    else:
        subfolder_names = subfolder_names_raw
        
    # Define the basic data structure
    data = {'problem': list(), 'dimensions': list(), 'results': list()}

    for subfolder in subfolder_names:
        # Extract the problem name and the number of dimensions
        subfolder_splitted_name = subfolder.split('-')
        problem_name = subfolder_splitted_name[0]
        dimensions = subfolder_splitted_name[1]

        # Store information about this subfolder
        data['problem'].append(problem_name)
        data['dimensions'].append(int(dimensions[:-1]))

        # Read all the iterations files contained in this subfolder
        temporal_full_path = os.path.join(main_folder, subfolder)

        # Iteration (in this case, operator) file names
        raw_file_names = [element for element in os.listdir(
            temporal_full_path) if not element.startswith('.')]

        # Sort the list of files based on their iterations
        file_names = sorted(raw_file_names, key=lambda x: int(x.split('-')[0]))

        # When using brute_force experiments, the last_step has no sense.
        if kind == 'brute_force':
            last_step = -1
            label_operator = 'operator_id'
            # Initialise iteration data
            file_data = {'operator_id': list(), 'performance': list(), 'statistics': list(), 'fitness': list()}
        elif kind == 'basic_metaheuristic':
            last_step = -1
            label_operator = 'operator_id'
            # Initialise iteration data
            file_data = {'operator_id': list(), 'performance': list(), 'statistics': list(),
                         'fitness': list(), 'hist_fitness': list()}  # !remove -> 'hist_fitness': list()
        elif kind == "unknown":
            last_step = int(file_names[-1].split('-')[0])
            label_operator = 'step'
            file_data = dict()
        elif kind == 'dynamic_metaheuristic':
            # TODO: Light conversion? - Verify compatibility with dynamic transfer learning
            last_step = int(file_names[-1].split('-')[0])
            label_operator = 'rep'
            file_data = {'rep': list(), 'hist_fitness': list(),
                         'encoded_solution': list(), 'performance': list()}
        elif kind == "dynamic_transfer_learning":
            last_step = int(file_names[-1].split('-')[0])
            label_operator = 'step'
            file_data = dict()
        elif kind == "static_transfer_learning":
            last_step = int(file_names[-1].split('-')[0])
            label_operator = 'step'
            file_data = dict(step=[], encoded_solution=[], performance=[], hist_fitness=[], hist_positions=[])
        else:
            last_step = int(file_names[-1].split('-')[0])
            label_operator = 'step'
            # Initialise iteration data
            file_data = {'step': list(), 'performance': list(), 'statistics': list(),
                         'encoded_solution': list(), 'hist_fitness': list(), 'hist_positions': list()}

        # Walk on the subfolder's files
        for file_name in tqdm(file_names, desc='{} {}, last={}'.format(problem_name, dimensions, last_step)):

            # Extract the iteration number and time
            operator_id = int(file_name.split('-')[0])

            # Read json file
            with open(temporal_full_path + '/' + file_name, 'r') as json_file:
                temporal_data = json.load(json_file)

            if kind == "dynamic_metaheuristic":
                file_data[label_operator].append(operator_id)
                file_data['encoded_solution'].append(temporal_data['encoded_solution'])
                file_data['hist_fitness'].append(temporal_data['best_fitness'])
                
            elif kind in ["unknown", "dynamic_transfer_learning"]:
                if len(file_data) != 0:
                    keys_to_use = list(file_data.keys())
                else:
                    keys_to_use = list(temporal_data.keys())

                if only_laststep and operator_id == last_step:
                    for field in keys_to_use:
                        file_data[field] = temporal_data[field]
                else:
                    if len(file_data) == 0:  # The first entering
                        # Read the available fields from the first file and create the corresponding lists
                        for field in keys_to_use:
                            file_data[field] = list()

                    # Fill the file_data
                    for field in list(keys_to_use):
                        file_data[field].append(temporal_data[field])

            else:
                # Store information in the corresponding variables
                file_data[label_operator].append(operator_id)
                file_data['performance'].append(temporal_data['performance'])
                if kind == 'brute_force':
                    file_data['statistics'].append(temporal_data['statistics'])
                    file_data['fitness'].append(temporal_data['fitness'])
                elif kind == 'basic_metaheuristic':
                    file_data['statistics'].append(temporal_data['statistics'])
                    file_data['fitness'].append(temporal_data['fitness'])
                    file_data['hist_fitness'].append(temporal_data['historical'])

                elif kind == "static_transfer_learning":
                    file_data['encoded_solution'].append(temporal_data['encoded_solution'])
                    file_data['performance'].append(temporal_data['performance'])

                    if only_laststep and operator_id == last_step:
                        file_data['hist_fitness'] = [x['fitness'] for x in temporal_data['details']['historical']]
                        file_data['hist_positions'] = [x['position'] for x in temporal_data['details']['historical']]
                    else:
                        file_data['hist_fitness'].append([x['fitness'] for x in temporal_data['details']['historical']])
                        file_data['hist_positions'].append([x['position'] for x in temporal_data['details']['historical']])

                else:
                    file_data['encoded_solution'].append(temporal_data['encoded_solution'])
                    file_data['statistics'].append(temporal_data['details']['statistics'])

                    # Only save the historical fitness values when operator_id is the largest one
                    if only_laststep and operator_id == last_step:
                        file_data['hist_fitness'] = [x['fitness'] for x in temporal_data['details']['historical']]
                        file_data['hist_positions'] = [x['position'] for x in temporal_data['details']['historical']]
                    else:
                        file_data['hist_fitness'].append([x['fitness'] for x in temporal_data['details']['historical']])
                        file_data['hist_positions'].append([x['position'] for x in temporal_data['details']['historical']])

            # Following information can be included but resulting files will be larger
            # file_data['fitness'].append(temporal_data['details']['fitness'])
            # file_data['positions'].append(temporal_data['details']['positions'])
        if kind == "dynamic_metaheuristic":
            best_fitness = [x[-1] for x in file_data['hist_fitness']]
            file_data['performance'] = st.iqr(best_fitness) + np.median(best_fitness)
        # Store results in the main data frame
        data['results'].append(file_data)

    # Save pre-processed data
    # save_json(data, file_name=main_folder.split('/')[0] + "/" + output_name)
    save_json(data, file_name=output_name)

    # Return only the data variable
    # return data


def df2dict(df):
    """
    Return a dictionary from a Pandas.dataframe.

    :param pandas.DataFrame df:
        Pandas' DataFrame.

    :return: dict.
    """
    df_dict = df.to_dict('split')
    return {df_dict['index'][x]: df_dict['data'][x] for x in range(len(df_dict['index']))}


def check_fields(default_dict, new_dict):
    """
    Return the dictionary with default keys and values updated by using the information of ``new_dict``

    :param dict default_dict:
        Dictionary with default values.
    :param dict new_dict:
        Dictionary with new values.
    :return: dict.
    """
    # Check if the entered variable has different values
    for key in list(set(default_dict.keys()) & set(new_dict.keys())):
        default_dict[key] = new_dict[key]
    return default_dict


def save_json(variable_to_save, file_name=None, suffix=None):
    """
    Save a variable composed with diverse types of variables, like numpy.

    :param any variable_to_save:
        Variable to save.
    :param str file_name: Optional.
        Filename to save the variable. If this is None, a random name is used. The default is None.
    :param str suffix: Optional.
        Prefix to put in the file_name. The default is None.

    :return: None
    """
    if file_name is None:
        file_name = 'autosaved-' + str(hex(random.randint(0, 9999)))

    suffix = '_' + suffix if suffix else ''

    # Create the new file
    with open('./{}{}.json'.format(file_name, suffix), 'w') as json_file:
        json.dump(variable_to_save, json_file, cls=NumpyEncoder)


def read_json(data_file):
    """
    Return data from a json file.

    :param str data_file:
        Filename of the json file.
    :return: dict or list.
    """
    with open(data_file, 'r') as json_file:
        data = json.load(json_file)

    # Return only the data variable
    return data


def merge_json(data_folder, list_of_fields=None):
    # data_folder = './data_files/raw/Sphere-50D'

    final_file_path = data_folder + '/' + data_folder.split('/')[-1] + '.csv'
    file_names = [element for element in os.listdir(data_folder) if ((not element.startswith('.'))
                                                                     and element.endswith('.json'))]

    temporal_pretable = list()

    for file_name in tqdm(file_names):

        temporal_data = read_json(data_folder + '/' + file_name)

        if (len(temporal_pretable) == 0) and (not list_of_fields):
            list_of_fields = list(temporal_data.keys())

        temporal_pretable.append({field: temporal_data[field] for field in list_of_fields})

    table = pd.DataFrame(temporal_pretable).to_csv(final_file_path)
    print('Merged file saved: {}'.format(final_file_path))



class NumpyEncoder(json.JSONEncoder):
    """
    Numpy encoder
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    # Import module for calling this code from command-line
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process results for a given experiment to make comparisons and visualisation with other experiments.'
    )
    parser.add_argument('experiment', 
                        metavar='experiment_filename', 
                        type=str, nargs=1, 
                        help='Name of finished experiment')
    parser.add_argument('output_filename', 
                        metavar='output_filename', 
                        type=str, nargs=1, 
                        default='experiment_results', 
                        help='Name of processed results file')
    parser.add_argument('experiment_kind', 
                        metavar='experiment_kind', 
                        type=str, nargs=1, 
                        default='dynamic_metaheuristic', 
                        help='Specify which type of experiment would be processed')    
    # preprocess_files(main_folder='data_files/raw/', output_name='brute_force')

    preprocess_files(main_folder='data_files/raw/', 
                     kind=parser.parse_args().experiment_kind[0], #'dynamic_metaheuristic', 
                     experiment=parser.parse_args().experiment[0], #'short_nn_mlp_30pop_performance_learning', 
                     output_name=parser.parse_args().output_filename[0])#'short_mlp_30pop_performance')
