# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""
import os
import numpy as np
from tools import save_json
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import scipy.stats as st
from experiment import read_config_file, create_task_list
import pjflacco as pf


# TODO: Avoid using pflacco (it is no longer maintained)


class Gorddo:
    def __init__(self, boundaries=None, dimensions=None, **kwargs):
        # TODO: Add feature of accepting specified list of functions, dimensions, and boundaries

        # Read the configuration file
        _, _, self.prob_config = read_config_file(**kwargs)

        # Set the forced boundaries and dimensions
        if boundaries:
            if isinstance(boundaries, tuple):
                lower, upper = boundaries

                # Hypercube problem domain
                if isinstance(lower, (float, int)) and isinstance(upper, (float, int)):
                    if dimensions:
                        if isinstance(dimensions, int) and (dimensions > 1):
                            self.num_dimensions = dimensions
                        else:
                            raise CharacteriserError('dimensions must be a integer greater than one!')
                    else:
                        self.num_dimensions = None

                    self.lower_boundaries = lower
                    self.upper_boundaries = upper

                # Parallelepiped problem domain
                else:
                    if len(lower) == len(upper):
                        if not dimensions:
                            dimensions = len(lower)
                        else:
                            if not isinstance(dimensions, int) or (1 > dimensions > len(lower)):
                                raise CharacteriserError(
                                    'dimensions must be a integer between 2 and len(boundaries[0])!')

                        self.num_dimensions = dimensions
                        self.lower_boundaries = np.array(lower[:dimensions])
                        self.upper_boundaries = np.array(upper[:dimensions])
                    else:
                        raise CharacteriserError('lower and upper boundaries must have the same length!')
            else:
                raise CharacteriserError('boundaries must be a tuple!')
        else:
            self.lower_boundaries = None
            self.upper_boundaries = None

            if dimensions:
                if isinstance(dimensions, int) and (dimensions > 1):
                    self.num_dimensions = dimensions
                else:
                    raise CharacteriserError('dimensions must be a integer greater than one!')
            else:
                self.num_dimensions = None

        self.problem_features = list()
        self.problem_names = list()
        self.problem_dimensions = list()

    def run(self, sampling_method='latin_hypercube'):
        # Create the combination of (problem, dimension) to be characterised (overwrite dimensions if specified)
        problems = create_task_list(self.prob_config['functions'],
                                    [self.num_dimensions] if self.num_dimensions else self.prob_config['dimensions'])
        num_problems = len(problems)

        # For each problem and dimension find the features
        for index, prob_dim in enumerate(problems, start=1):
            problem_string, num_dimensions = prob_dim

            # Call the problem to characterise
            problem_object = bf.choose_problem(problem_string, num_dimensions)

            # Overwrite the boundaries of the problem domain
            if self.lower_boundaries:  # also self.upper_boundaries
                problem_object.set_search_range(self.lower_boundaries, self.upper_boundaries)

            # Create a characteriser object
            chsr = Characteriser(sampling_method)

            # Mark start the characterising procedure
            print('Characterising {}-{}D...'.format(problem_string, num_dimensions))

            # Calculate the features
            self.problem_features.append(chsr.characterise(problem_object))
            self.problem_names.append(problem_string)
            self.problem_dimensions.append(num_dimensions)

            # Mark end the characterising procedure
            print('DONE! [{}/{}]'.format(index, num_problems))

    def save_results(self, file_name=None, folder_name=None):

        folder_name = 'characteristics/' if not folder_name else folder_name

        # Verify if the path exists
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        if not (len(self.problem_features) == 0):
            save_json(dict(
                name=self.problem_names,
                dimensions=self.problem_dimensions,
                features=self.problem_features
            ), folder_name + file_name, suffix='features')
        else:
            raise CharacteriserError('Features variable is empty, len = 0')

        """
        Set the problem domain using boundaries information as well as dimensions, if so.

        :param tuple boundaries:
            Domain boundaries for the problem domain, they must be given in a tuple such as (`lower_boundary`,
            `upper_boundary`), since `lower_boundary` and `upper_boundary` are either int|float or lists of
            int|float.
            Depending of the nature of these boundaries certain behaviours are considered:
                - `lower_boundary` and `upper_boundary` are int|float -> Problem domain is given by a hypercube so
                    `dimensions` are mandatory.
                - `lower_boundary` and `upper_boundary` are array-like (lists) -> Problem domain is given by a
                    parallelepiped so `dimensions` are optional (this value is inferred from `lower_boundary`).
                    If `dimensions` are specified, its value must be between 2 and `len(lower_boundary)`. Therefore,
                    if this value is lower than `len(lower_boundary)`, it is assumed that the entered boundaries are
                    quite general so, for such a case, `dimensions` is highly relevant and is used to reshape the
                    boundary lists, i. e., `new_boundary = old_boundary[:dimensions]`.

        :param int dimensions:
            Number of dimensions to modify the internal value. Default is None.

        :return: None
        """


class Characteriser:
    def __init__(self, sampling_method='latin_hypercube'):
        self.sampling_method = sampling_method

        self.num_blocks = 3
        self.is_minimising = True

        self.available_features = pf.list_available_feature_sets(False, False)

        # self.bandwidth = 1
        # self.kde_samples = 1000
        # self.num_repetitions = 30
        # self.normalised_boundaries = True

        # self.levy_walk_initial = 'rand'
        # self.levy_walk_alpha = 0.5
        # self.levy_walk_beta = 1.0

        # self.position_samples = None
        # self.fitness_values = None

    # TODO: Generalise for multiple feature suites
    def characterise(self, problem_object, samples=None):

        # TODO: Add other kind of problem definition (same for evaluation)
        lower_boundaries = problem_object.min_search_range
        upper_boundaries = problem_object.max_search_range
        num_dimensions = problem_object.variable_num
        # If the problem object is a string, assume that it refers to an benchmark_func's object
        # if isinstance(problem_object, bf.BasicProblem):
        # Read boundaries and determine the span and centre : high level -> self. _boundaries
        # lower_boundaries = self.lower_boundaries if self.lower_boundaries else problem_object.min_search_range
        # upper_boundaries = self.upper_boundaries if self.upper_boundaries else problem_object.max_search_range
        # span_boundaries = upper_boundaries - lower_boundaries
        # centre_boundaries = (upper_boundaries + lower_boundaries) / 2.
        # else:
        #     raise CharacteriserError('Problem object not recognised!')

        # Read the number of dimensions
        # num_dimensions = self.num_dimensions if self.num_dimensions else len(lower_boundaries)

        # Update the number of observations
        num_samples = samples if samples else 50 * num_dimensions

        # Generate the samples
        # TODO: Add more methods for generating samples
        if self.sampling_method == 'latin_hypercube':  # available with pflacco
            position_samples = pf.create_initial_sample(n_obs=num_samples, dim=num_dimensions, type='lhs',
                                                        lower_bound=lower_boundaries, upper_bound=upper_boundaries)
        # elif self.sampling_method == 'levy_walk':
        #     self.position_samples = self._levy_walk(self.levy_walk_initial, self.num_samples,
        #                                             self.levy_walk_alpha, self.levy_walk_beta)
        else:
            position_samples = lower_boundaries + (upper_boundaries - lower_boundaries) * np.random.random_sample(
                (num_samples, num_dimensions))

        # Evaluate them in the function
        fitness_values = problem_object.get_function_values(position_samples)

        # Create the feature object
        feature_object = pf.create_feature_object(x=position_samples, y=fitness_values, minimize=self.is_minimising,
                                                  lower=lower_boundaries, upper=upper_boundaries)
        # blocks=self.num_blocks)
        # fun=lambda x: problem_object.get_function_values(x))

        # Calculate all the features
        # TODO: Revise how to calculate specific features (probably we need to modify pflacco)
        # feature_values = pf.calculate_features(feat_object=feature_object)
        feature_values = dict()
        for feature_set_name in self.available_features:
            feature_values.update(pf.calculate_feature_set(feat_object=feature_object, set_name=feature_set_name))

        return feature_values

    # features.append(feature_values)

    # def length_scale(self, function=None, bandwidth_mode='silverman_rule', samples=None, kde_samples=1000):
    #     # Samples from the estimated pde
    #     self.kde_samples = kde_samples
    #
    #     # Initialise the sample positions and their fitness
    #     if ((self.position_samples is None) | (self.fitness_values is None)) & (function is not None):
    #         self.initialise(function, samples)
    #
    #     # Determine the length scale
    #     indices_1 = np.random.permutation(len(self.fitness_values))
    #     indices_2 = np.array([*indices_1[1:], indices_1[0]])
    #
    #     length_scale = (np.abs(self.fitness_values[indices_1] - self.fitness_values[indices_2]) / np.linalg.norm(
    #         self.position_samples[indices_1, :] - self.position_samples[indices_2, :], axis=1)).reshape(-1, 1)
    #
    #     # Estimate the bandwidth
    #     if not isinstance(bandwidth_mode, str):
    #         self.bandwidth = bandwidth_mode
    #     else:
    #         if bandwidth_mode == 'exhaustive':
    #             order = int(np.ceil(np.log10(np.std(length_scale))))
    #             coarse_grid = GridSearchCV(KernelDensity(),
    #                                        {'bandwidth': np.logspace(order / 2 - 3, order / 2 + 3, 25)}, cv=3)
    #             first_approach = coarse_grid.fit(length_scale).best_estimator_.bandwidth
    #             fine_grid = GridSearchCV(KernelDensity(), {'bandwidth':
    #                 np.linspace(0.5 * first_approach, 2 * first_approach, 50)}, cv=3)
    #             self.bandwidth = fine_grid.fit(length_scale).best_estimator_.bandwidth
    #         elif bandwidth_mode == 'scott_rule':
    #             self.bandwidth = 1.06 * np.std(length_scale) * np.power(self.num_samples, -1/5)
    #         elif bandwidth_mode == 'silverman_rule':
    #             self.bandwidth = 0.9 * np.min([np.std(length_scale), st.iqr(length_scale)/1.34]) * \
    #                              np.power(self.num_samples, -1/5)
    #         else:
    #             self.bandwidth = None
    #
    #     # Estimate the distribution function
    #     pdf_xvalues = np.linspace(0.9 * length_scale.min(), 1.1 * length_scale.max(), self.kde_samples).reshape(-1, 1)
    #     pdf_fvalues = np.exp(KernelDensity(bandwidth=self.bandwidth).fit(length_scale).score_samples(pdf_xvalues))
    #
    #     # Get statistics from raw length_scale values
    #     dst = st.describe(length_scale)
    #
    #     # Determine the entropy metric
    #     entropy_value = (pdf_xvalues[1] - pdf_xvalues[0]) * st.entropy(pdf_fvalues, base=2)
    #
    #     # Return a dictionary with all the information
    #     return dict(nob=dst.nobs,
    #                 raw=length_scale,
    #                 Min=dst.minmax[0],
    #                 Max=dst.minmax[1],
    #                 Avg=dst.mean,
    #                 Std=np.std(length_scale),
    #                 Skw=dst.skewness,
    #                 Kur=dst.kurtosis,
    #                 IQR=st.iqr(length_scale),
    #                 Med=np.median(length_scale),
    #                 MAD=st.median_abs_deviation(length_scale),
    #                 KDE_bw=self.bandwidth,
    #                 PDF_fx=pdf_fvalues,
    #                 PDF_xs=pdf_xvalues,
    #                 Entropy=entropy_value)
    #
    # @staticmethod
    # def _normalise_vector(vector):
    #     return vector / np.max([np.linalg.norm(vector), 1e-23])

    # def _levy_walk(self, initial_position, num_steps=1000, alpha=0.5, beta=1.0):
    #
    #     # Initial position and all the positions are normalised between -1 and 1
    #     if initial_position == 'rand':
    #         initial_position = np.random.uniform(-1, 1, self.num_dimensions)
    #     else:
    #         if not len(initial_position) == self.num_dimensions:
    #             raise CharacteriserError('Provide a proper initial position')
    #
    #     # Initialise the output matrix
    #     positions = [initial_position]
    #
    #     # Start the loop for all the steps
    #     while len(positions) <= num_steps + 1:
    #         # Get the Levy-distributed step and a point in the hyper-sphere surface
    #         new_position = positions[-1] + st.levy_stable.rvs(
    #             alpha, beta, size=self.num_dimensions) * self._normalise_vector(
    #             np.random.randn(self.num_dimensions))
    #
    #         # Check if this position is within the domain and register it
    #         if (new_position > -1.0).all() & (new_position < 1.0).all():
    #             positions.append(new_position)
    #
    #     return np.array(positions)


class CharacteriserError(Exception):
    """
    Simple CharacteriserError to manage exceptions.
    """
    pass


if __name__ == '__main__':
    import benchmark_func as bf
    import matplotlib.pyplot as plt

    # results_all = []
    #
    # for problem_str in bf.__all__:
    #     problem = eval('bf.' + problem_str + '(2)')
    #
    #     chsr = Characteriser()
    #     results_all.append(chsr.length_scale(problem, bandwidth_mode='silverman_rule')['Entropy'])
    #
    #     print('Evaluated ' + problem_str + '...')
    #
    # plt.semilogy([res + 1 for res in results_all]), plt.show()

    # problem = bf.Sphere(2)
    # problem.set_search_range(-5, 5)
    # chsr = Characteriser()
    # results = chsr.characterise(problem)

    gdd = Gorddo(prob_config={'dimensions': [2, 5, 10, 20, 30, 40, 50], 'functions': bf.__all__})
    gdd.run()
    gdd.save_results('all')

    # results = chsr.length_scale(problem, bandwidth_mode='exhaustive')
    # plt.hist(results['raw'], density=True, bins=100), plt.plot(results['PDF_xs'], results['PDF_fx']), plt.show()

    # print(results['Entropy'])

# def fast_univariate_bandwidth_estimate_STEPI(
#         num_points, source_points,  accuracy=1e-3):
#
#     # Normalise data to the unit interval
#     normalised_source_points = (source_points - np.min(source_points)) / np.max(source_points)
#
#     # Estimate the standard deviation of data
#     sigma = np.std(normalised_source_points)
#
#     # Density functional Phi_6 and Phi_8 via the normal scale rule
#     phi6 = -15 * np.power(sigma, -7) / (16 * np.sqrt(np.pi))
#     phi8 = -105 * np.power(sigma, -9) / (32 * np.sqrt(np.pi))
#
#     g1 = np.power(-6 / (np.sqrt(2 * np.pi) * phi6 * num_points), 1 / 7)
#     g1 = np.power(30 / (np.sqrt(2 * np.pi) * phi8 * num_points), 1 / 9)
