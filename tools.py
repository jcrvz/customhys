"""
Created on Sat Feb 22, 2020

@author: jcrvz (jcrvz.github.io)
"""
import os
import json
from subprocess import call
from hyperheuristic import NumpyEncoder
from tqdm import tqdm

def printmsk(var, level=1, name=None):
    """
    Print the meta-skeleton of a variable with nested variables.

    :param var: any, variable to inspect.
    :param level: int (optional), level of the variable to inspect. Default: 1.
    :param name: name (optional), name of the variable to inspect. Default: None.
    :return: None

    Example:

    >>> variable = {"par0": [1, 2, 3, 4, 5, 6],
            "par1": [1, 'val1', 1.23],
            "par2": -4.5,
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
    """
    # Parent inspection
    parent_type = var.__class__.__name__
    var_name = "" if name is None else name + " = "
    print('|  ' * (level - 1) + '|-- ' + var_name + "{", end="")

    if hasattr(var, '__len__') and not (parent_type in ['str', 'ndarray']):
        print("{}: {}".format(parent_type, len(var)) + "}")

        # If is it a dictionary
        if parent_type == 'dict':
            for key, val in var.items():
                printmsk(val, level + 1, key)
        elif parent_type in ['list', 'tuple']:
            # Get a sample: first 10 elements (if the list is too long)
            if len(var) > 10:
                var = var[:10]

            # If all the elements has same type, then show an example
            if len(set([val.__class__.__name__ for val in var])) == 1:
                printmsk(var[0], level + 1, "0")
                print(':  ' * (level + 1))
            else:
                for id in range(len(var)):
                    printmsk(var[id], level + 1, str(id))
    else:
        if parent_type == 'ndarray':
            dimensions = " x ".join([str(x) for x in var.shape])
            print("{}: {}".format(parent_type, dimensions) + "}")
        else:
            print("{}".format(parent_type) + "}")


def listfind(values, val):
    """
    Find indices of a list corresponding to a value.

    :param values: list, a list to analyse.
    :param val: any, element to find in the list.
    :return: a list of indices.
    """
    return [i for i in range(0, len(values)) if values[i] == val]


def revise_results(main_folder='data_files/raw/'):
    """
    Revise a folder with subfolders, and check if there are subfolder repeated,
    in name, to merge.

    :param main_folder: root to analyse
    :return: None
    """
    raw_folders = [element for element in os.listdir(main_folder)
                   if not element.startswith('.')]
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
                call(['cp', '-a', main_folder + folders_with_date[index] + '/*',
                      destination_folder])  # Linux
                # Rename the copied folder with prefix '_to_delete_'
                call(['mv', main_folder + folders_with_date[index],
                      main_folder + ".to_delete-" + folders_with_date[index]])
                print("Merged '{}' into '{}'!".format(
                    folders_with_date[index], folders_with_date[indices[0]]))


def preprocess_bruteforce_files(main_folder='data_files/raw/'):
    # Get folders and exclude hidden ones
    raw_folders = [element for element in os.listdir(main_folder)
                   if not element.startswith('.')]

    # Sort subfolder names by problem name & dimensions
    subfolder_names = sorted(raw_folders,
                             key=lambda x: int(x.split('-')[1].strip('D')))

    # Define the basic data structure
    data = {'problem': list(), 'dimensions': list(), 'results': list()}

    for subfolder in subfolder_names:
        # Extract the problem name and the number of dimensions
        problem_name, dimensions, date_str = subfolder.split('-')

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

        # Initialise iteration data with same field as files
        # details only contains fitness values and positions
        # file_data = {'operator_id': list(), 'performance': list(),
        #                   'fitness': list(), 'positions': list()}
        file_data = {'operator_id': list(), 'performance': list(),
                     'statistics': list()}

        # Walk on subfolders' files
        for file_name in tqdm(file_names,
            desc='{} {}'.format(problem_name, dimensions)):

            # Extract the iteration number and time
            operator_id = int(file_name.split('-')[0])

            # Read json file
            with open(temporal_full_path + '/' + file_name, 'r') as json_file:
                temporal_data = json.load(json_file)

            # Store information in the correspoding variables
            file_data['operator_id'].append(operator_id)
            file_data['performance'].append(temporal_data['performance'])
            # file_data['fitness'].append(temporal_data['details']['fitness'])
            # file_data['positions'].append(temporal_data['details']['positions'])
            file_data['statistics'].append(temporal_data['details']['statistics'])

        # Store results in the main data frame
        data['results'].append(file_data)

    # Save pre-processed data
    with open(main_folder.split('/')[0] + "/brute-force-data.json", 'w') as json_file:
        json.dump(data, json_file, cls=NumpyEncoder)

    # Return only the data variable
    return data


if __name__ == '__main__':
    processed_data = preprocess_bruteforce_files()

# .to_delete-HyperEllipsoid-50D-02_22_2020
# .to_delete-Mishra7-40D-02_22_2020
# .to_delete-Perm-30D-02_22_2020
# .to_delete-Rastrigin-20D-02_22_2020
# .to_delete-Schubert-10D-02_22_2020
# .to_delete-Schwefel-5D-02_22_2020
# .to_delete-Schwefel221-2D-02_22_2020
# .to_delete-Schwefel223-50D-02_23_2020
# .to_delete-WWavy-40D-02_23_2020
# .to_delete-Zakharov-2D-02_24_2020

