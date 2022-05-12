# -*- coding: utf-8 -*-
"""
This module contains the main experiments performed using the current framework.

Created on Mon Sep 30 13:42:15 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

from customhys import hyperheuristic as hyp
from customhys import operators as Operators
from customhys import benchmark_func as bf
from customhys import tools as tl
import multiprocessing
from os import path


# %% PREDEFINED CONFIGURATIONS
# Use configuration files instead of predefined dictionaries

# Configuration dictionary for experiments
ex_configs = [
    {'experiment_name': 'demo_test', 'experiment_type': 'default', 'heuristic_collection_file': 'default.txt',
     'weights_dataset_file': 'operators_weights.json'},  # 0 - Default
    {'experiment_name': 'brute_force', 'experiment_type': 'brute_force',
     'heuristic_collection_file': 'default.txt'},  # 1 - Brute force
    {'experiment_name': 'basic_metaheuristics', 'experiment_type': 'basic_metaheuristics',
     'heuristic_collection_file': 'basicmetaheuristics.txt'},  # 2 - Basic metaheuristics
    {'experiment_name': 'short_test1', 'experiment_type': 'default', 'heuristic_collection_file': 'default.txt',
     'weights_dataset_file': 'operators_weights.json'},  # 3 - Short collection
    {'experiment_name': 'short_test2', 'experiment_type': 'default',
     'heuristic_collection_file': 'default.txt'},  # 4 - Short collection +
    {'experiment_name': 'long_test', 'experiment_type': 'default', 'heuristic_collection_file': 'test-set-21.txt',
     'auto_collection_num_vals': 21}  # 5 - Long collection
]

# Configuration dictionary for hyper-heuristics
hh_configs = [
    {'cardinality': 3, 'num_replicas': 30},  # 0 - Default
    {'cardinality': 1, 'num_replicas': 30},  # 1 - Brute force
    {'cardinality': 1, 'num_replicas': 30},  # 2 - Basic metaheuristic
    {'cardinality': 3, 'num_replicas': 50},  # 3 - Short collection
    {'cardinality': 5, 'num_replicas': 50},  # 4 - Short collection +
    {'cardinality': 3, 'num_replicas': 50}   # 5 - Long collection
]

# Configuration dictionary for problems
pr_configs = [
    {'dimensions': [2, 5], 'functions': ['<choose_randomly>']},  # 0 - Default
    {'dimensions': [2, 5, *range(10, 50 + 1, 10)], 'functions': bf.__all__},    # 1 - Brute force
    {'dimensions': [2, 5, *range(10, 50 + 1, 10)], 'functions': bf.__all__},    # 2 - Basic metaheuristic
    {'dimensions': [2, 5, *range(10, 50 + 1, 10)], 'functions': bf.__all__},    # 3 - Short collection
    {'dimensions': [2, 5, *range(10, 50 + 1, 10)], 'functions': bf.__all__},    # 4 - Short collection +
    {'dimensions': [2, 5, *range(10, 50 + 1, 10)], 'functions': bf.__all__},    # 5 - Long collection
]


# %% EXPERIMENT CLASS

# _ex_config_demo = {'experiment_name': 'demo_test', 'experiment_type': 'default',
#                    'heuristic_collection_file': 'default.txt', 'weights_dataset_file': 'operators_weights.json'}
# _hh_config_demo = {'cardinality': 3, 'num_replicas': 30}
# _pr_config_demo = {'dimensions': [2, 5], 'functions': [bf.__all__[hyp.np.random.randint(0, len(bf.__all__))]]}

class Experiment:
    """
    Create an experiment using certain configurations.
    """

    def __init__(self, config_file=None, exp_config=None, hh_config=None, prob_config=None):
        """
        Initialise the experiment object.

        :param str config_file:
            Name of the configuration JSON file with the configuration dictionaries: exp_config, hh_config, and
            prob_config. If only the filename is provided, it is assumed that such a file is in the directory
            './exconf/'. Otherwise, the full path must be entered. The default value is None.

        :param dict exp_config:
            Configuration dictionary related to the experiment. Keys and default values are listed as follows:

            'experiment_name':              'test',         # Name of the experiment
            'experiment_type':              'default',      # Type: 'default', 'brute_force', 'basic_metaheuristics'
            'heuristic_collection_file':    'default.txt',  # Heuristic space located in /collections/
            'weights_dataset_file':         None,           # Weights or probability distribution of heuristic space
            'use_parallel':                 True,           # Run the experiment using a pool of processors
            'parallel_pool_size':           None,           # Number of processors available, None = Default
            'auto_collection_num_vals':     5               # Number of values for creating an automatic collection

            **NOTE 1:** 'experiment_type': 'default' or another name mean hyper-heuristic.
            **NOTE 2:** If the collection does not exist and it is not a reserved one ('default.txt', 'automatic.txt',
            'basicmetaheuristics.txt', 'test_collection'), then an automatic heuristic space is generated with
            ``Operators.build_operators`` with 'auto_collection_num_vals' as ``num_vals`` and
            'heuristic_collection_file' as ``filename``.
            **NOTE 3:** # 'weights_dataset_file' must be determined in a pre-processing step. For the 'default'
            heuristic space, it is provided 'operators_weights.json'.

        :param dict hh_config:
            Configuration dictionary related to the hyper-heuristic procedure. Keys and default values are listed as
            follows:

            'cardinality':                      3,          # Maximum cardinality used for building metaheuristics
            'num_agents':                       30,         # Population size employed by the metaheuristic
            'num_iterations':                   100,        # Maximum number of iterations used by the metaheuristic
            'num_replicas':                     50,         # Number of replicas for each metaheuristic implemented
            'num_steps':                        100,        # * Number of steps that the hyper-heuristic performs
            'max_temperature':                  200,        # * Initial temperature for HH-Simulated Annealing
            'stagnation_percentage':            0.3,        # * Percentage of stagnation used by the hyper-heuristic
            'cooling_rate':                     0.05        # * Cooling rate for HH-Simulated Annealing

            **NOTE 4:** Keys with * correspond to those that are only used when ``exp_config['experiment_type']`` is
            neither 'brute_force' or 'basic_metaheuristic'.

        :param dict prob_config:
            Configuration dictionary related to the problems to solve. Keys and default values are listed as follows:

            'dimensions':       [2, 5, 10, 20, 30, 40, 50], # List of dimensions for the problem domains
            'functions':        bf.__all__,                 # List of function names of the optimisation problems
            'is_constrained':   True                        # True if the problem domain is hard constrained

        :return: None.

        """
        self.exp_config, self.hh_config, self.prob_config = read_config_file(config_file, exp_config, hh_config,
                                                                             prob_config)

        # Check if the heuristic collection exists
        if not path.isfile('./collections/' + self.exp_config['heuristic_collection_file']):
            # If the name is a reserved one. These files cannot be not created automatically
            if exp_config['heuristic_collection_file'] in ['default.txt', 'automatic.txt', 'basicmetaheuristics.txt',
                                                           'test_collection']:
                raise ExperimentError('This collection name is reserved and cannot be created automatically!')
            else:
                Operators.build_operators(Operators.obtain_operators(
                    num_vals=exp_config['auto_collection_num_vals']),
                    file_name=exp_config['heuristic_collection_file'].split('.')[0])
                self.exp_config['weights_dataset_file'] = None

        # Check if the weights dataset not exist or required
        if self.exp_config['weights_dataset_file'] and (
                self.exp_config['experiment_type'] not in ['brute_force', 'basic_metaheuristics']):
            if path.isfile('collections/' + self.exp_config['weights_dataset_file']):
                self.weights_data = tl.read_json('collections/' + self.exp_config['weights_dataset_file'])
            else:
                raise ExperimentError('A valid weights_dataset_file must be provided in exp_config')
        else:
            self.weights_data = None

    def run(self):
        """
        Run the experiment according to the configuration variables.

        :return: None
        """
        # TODO: Create a task log for prevent interruptions
        # Create a list of problems from functions and dimensions combinations
        # all_problems = [(x, y) for x in self.prob_config['functions'] for y in self.prob_config['dimensions']]
        all_problems = create_task_list(self.prob_config['functions'], self.prob_config['dimensions'])

        # Check if the experiment will be in parallel
        if self.exp_config['use_parallel']:
            pool = multiprocessing.Pool(self.exp_config['parallel_pool_size'])
            pool.map(self._simple_run, all_problems)
        else:
            for prob_dim in all_problems:
                self._simple_run(prob_dim)

    def _simple_run(self, prob_dim):
        """
        Perform a single run, i.e., for a problem and dimension combination

        :param tuple prob_dim:
            Problem name and dimensionality ``(function_string, num_dimensions)``

        :return: None.
        """
        # Read the function name and the number of dimensions
        function_string, num_dimensions = prob_dim

        # Message to print and to store in folders
        label = '{}-{}D-{}'.format(function_string, num_dimensions, self.exp_config['experiment_name'])

        # Get and format the problem
        # problem = eval('bf.{}({})'.format(function_string, num_dimensions))
        problem = bf.choose_problem(function_string, num_dimensions)
        problem_to_solve = problem.get_formatted_problem(self.prob_config['is_constrained'],
                                                         self.prob_config['features'])

        # Read the particular weights array (if so)
        weights = self.weights_data[str(num_dimensions)][problem.get_features(fmt='string', wrd='1')] \
            if self.weights_data else None

        # Call the hyper-heuristic object
        hh = hyp.Hyperheuristic(heuristic_space=self.exp_config['heuristic_collection_file'],
                                problem=problem_to_solve, parameters=self.hh_config,
                                file_label=label, weights_array=weights)

        # Run the HH according to the specified type
        if self.exp_config['experiment_type'] in ["brute_force", 'bf']:
            hh.brute_force()
        elif self.exp_config['experiment_type'] == ["basic_metaheuristics", 'bmh']:
            hh.basic_metaheuristics()
        elif self.exp_config['experiment_type'] in ["online_learning", 'dynamic']:
            _ = hh.solve('dynamic')
        elif self.exp_config['experiment_type'] in ["transfer_learning"]:
            _ = hh.solve()
        elif self.exp_config['experiment_type'] in ["machine_learning", 'neural_network']:
            _ = hh.solve('neural_network')
        else:  # 'static_run'
            _ = hh.solve('static')

        # TODO: Add pre-label to know the current status
        print(label + ' done!')


class ExperimentError(Exception):
    """
    Simple ExperimentError to manage exceptions.
    """
    pass


def read_config_file(config_file=None, exp_config=None, hh_config=None, prob_config=None):
    """
    Return the experimental (`exp_config`), hyper-heuristic (`hh_config`), problem (`prob_config`) configuration
    variables from `config_file`, if it is supplied. Otherwise, use the `exp_config`, `hh_config`, and `prob_config`
    inputs. If there is no input, then assume the default values for these three configuration variables. Further
    information about these variables can be found in the Experiment class's `__doc__`.
    """

    # If there is a configuration file
    if config_file:
        # Adjustments
        directory, filename = path.split(config_file)
        if directory == '':
            directory = './exconf/'  # Default directory
        basename, extension = path.splitext(filename)
        if extension not in ['.json', '']:
            raise ExperimentError("Configuration file must be JSON")

        # Existence verification and load
        full_path = path.join(directory, basename + '.json')
        if path.isfile(full_path):
            all_configs = tl.read_json(full_path)

            # Load data from json file
            exp_config = all_configs['ex_config']
            hh_config = all_configs['hh_config']
            prob_config = all_configs['prob_config']
        else:
            raise ExperimentError(f"File {full_path} does not exist!")
    else:
        if exp_config is None:
            exp_config = dict()
        if hh_config is None:
            hh_config = dict()
        if prob_config is None:
            prob_config = dict()

    # Load the default experiment configuration and compare it with exp_cfg
    exp_config = tl.check_fields(
        {
            'experiment_name': 'test',
            'experiment_type': 'default',  # 'default' -> hh, 'brute_force', 'basic_metaheuristics'
            'heuristic_collection_file': 'default.txt',
            'weights_dataset_file': None,  # 'operators_weights.json',
            'use_parallel': True,
            'parallel_pool_size': None,  # Default
            'auto_collection_num_vals': 5
        }, exp_config)

    # Load the default hyper-heuristic configuration and compare it with hh_cfg
    hh_config = tl.check_fields(
        {
            'cardinality': 3,
            'num_agents': 30,
            'num_iterations': 100,
            'num_replicas': 50,
            'num_steps': 200,
            'max_temperature': 1,
            'min_temperature': 1e-6,
            'stagnation_percentage': 0.37,
            'cooling_rate': 1e-3,
            'cardinality_min': 1,
            'repeat_operators': True,
            'as_mh': True,
            'verbose': False,
            'trial_overflow': True,
            'learnt_dataset': None,
            'allow_weight_matrix': True,
            'learning_portion': 0.37,
            'solver': 'static',
            'tabu_idx': None,
            'model_params': None
        }, hh_config)

    # Load the default problem configuration and compare it with prob_config
    prob_config = tl.check_fields(
        {
            'dimensions': [2, 5, *range(10, 50 + 1, 10)],
            'functions': bf.__all__,
            'is_constrained': True,
            'features': ['Differentiable', 'Separable', 'Unimodal']
        }, prob_config)

    # Check if there is a special case of function name: <choose_randomly>
    prob_config['functions'] = [
        bf.__all__[hyp.np.random.randint(0, len(bf.__all__))] if fun in ['<choose_randomly>', '<random>'] else fun
        for fun in prob_config['functions']]

    return exp_config, hh_config, prob_config


# TODO: Create a task function that read which config variable is a list
def create_task_list(function_list, dimension_list):
    """
    Return a list of combinations (in tuple form) for problems from functions and dimensions.

    :param list function_list:
        List of functions from the `benchmark_func` module.

    :param list dimension_list:
        List of dimensions considered for each one of these functions.

    :return: list of tuples
    """
    # TODO: Create a task list for prevent interruptions
    return [(x, y) for x in function_list for y in dimension_list]


def __print_dic(dict_var):
    for key, val in dict_var.items():
        print(f"--> \"{key}\"", f"{val}", sep=": ", end=",\n")


# %% Auto-run
if __name__ == '__main__':
    # Import module for calling this code from command-line
    import argparse
    import os
    from tools import preprocess_files

    DATA_FOLDER = "./data_files/raw"
    OUTPUT_FOLDER = "./data_files/exp_output"

    # Only one argument is allowed: the code
    parser = argparse.ArgumentParser(
        description="Run certain experiment, default experiment is './exconf/demo.json'")
    parser.add_argument('-b', '--batch', action='store_true', help="carry out a batch of experiments")
    parser.add_argument('exp_config', metavar='config_filename', type=str, nargs='?', default='demo',
                        help='''Name of the configuration file in './exconf/' or its full path. Only JSON files.
                                If --batch flag is given, it is assumed that the entered file contains a list of all the
                                paths of experiment files (JSON) to carry out. It would be read as plain text.''')
    # choices = [x.split('.')[0] for x in listdir('./exconf') if x.split('.')[1] == 'json'],
    # exp_filename = list(vars(parser.parse_args()).values())[0]

    args = parser.parse_args()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if args.batch:
        with open(args.exp_config) as configs:
            exp_filenames = configs.read()
        exp_filenames = [filename.strip() for filename in exp_filenames.splitlines()]
    else:
        exp_filenames = [args.exp_config]

    for exp_filename in exp_filenames:
        tail_message = f" - ({exp_filenames.index(exp_filename)+1}/{len(exp_filenames)})" + "\n" + ("=" * 50) \
            if args.batch else ""

        print(f"\nRunning {exp_filename.split('.')[0]}" + tail_message)

        # Create the experiment to runs
        exp = Experiment(config_file=exp_filename)

        # print("* Experiment configuration: \n", "-" * 30 + "\n", json.dumps(exp.prob_config, indent=2, default=str))
        # print("* Hyper-heuristic configuration: \n", "-" * 30 + "\n", json.dumps(exp.hh_config, indent=2, default=str))
        # print("* Problem configuration: \n", "-" * 30 + "\n", json.dumps(exp.prob_config, indent=2, default=str))

        # print("\n* Experiment configuration:")
        # __print_dic(exp.prob_config)
        # print("\n* Hyper-heuristic configuration:")
        # __print_dic(exp.hh_config)
        # print("\n* Problem configuration:")
        # __print_dic(exp.prob_config)

        # Run the experiment et voilà
        exp.run()

        # After run, it preprocesses all the raw data
        print(f"\nPreprocessing {exp_filename.split('.')[0]}" + tail_message)
        preprocess_files("data_files/raw/",
                         kind=exp.hh_config["solver"],
                         output_name=OUTPUT_FOLDER + "/" + exp.exp_config["experiment_name"])

        # Rename the raw folder to raw-$exp_name$
        print(f"\nChanging folder name of raw results...")
        os.rename(DATA_FOLDER, DATA_FOLDER + "-" + exp.exp_config["experiment_name"])

    print("\nExperiment(s) finished!")
