"""
Created on Sat Feb 22, 2020

@author: jcrvz (jcrvz.github.io)
"""
import os
from subprocess import call


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

    if hasattr(var, '__len__') and (not isinstance(var, str)):
        print("{}: {}".format(parent_type, len(var)) + "}")

        # If is it a dictionary
        if isinstance(var, dict):
            for key, val in var.items():
                printmsk(val, level + 1, key)
        elif isinstance(var, (list, tuple)):
            # If all the elements has same type, then show an example
            if len(set([val.__class__.__name__ for val in var])) == 1:
                printmsk(var[0], level + 1, "0")
                print(':  ' * (level + 1))
            else:
                for id in range(len(var)):
                    printmsk(var[id], level + 1, str(id))
    else:
        print("{}".format(parent_type) + "}")


def listfind(values, val):
    return [i for i in range(0, len(values)) if values[i] == val]


def revise_results(main_folder='data_files/raw/'):
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
                call(['cp', '-a', main_folder + folders_with_date[index] + '/',
                      destination_folder])  # Linux
                # Rename the copied folder with prefix '_to_delete_'
                call(['mv', main_folder + folders_with_date[index],
                      main_folder + ".to_delete-" + folders_with_date[index]])
                print("Merged '{}' into '{}'!".format(
                    folders_with_date[index], folders_with_date[indices[0]]))

if __name__ == '__main__':
    revise_results()