# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import scipy.stats as st


class Characteriser():

    # _normal_scaling_factor = np.power(1 / (4 * np.pi), 1/10)

    def __init__(self):
        # Define the parameters
        self.bandwidth = 1
        self.kde_samples = 1000
        self.num_repetitions = 30
        self.normalised_boundaries = True
        self.num_dimensions = None

        self.num_samples = 1000
        self.sampling_method = 'levy_walk'
        self.levy_walk_initial = 'rand'
        self.levy_walk_alpha = 0.5
        self.levy_walk_beta = 1.0

        self.position_samples = None
        self.fitness_values = None

    # Initialise the sampling method and the function evaluation
    def initialise(self, function, samples=None):
        # Read boundaries and determine the span and centre
        lower_boundaries = function.max_search_range
        upper_boundaries = function.min_search_range
        span_boundaries = upper_boundaries - lower_boundaries
        centre_boundaries = (upper_boundaries + lower_boundaries) / 2.

        # Read the number of dimensions
        self.num_dimensions = len(centre_boundaries)

        # Generate the samples
        # TODO: Add more methods for generating samples
        if samples is None:
            if self.sampling_method == 'levy_walk':
                self.position_samples = self._levy_walk(self.levy_walk_initial, self.num_samples,
                                                        self.levy_walk_alpha, self.levy_walk_beta)

        # Evaluate them in the function
        self.fitness_values = self._evaluate_positions(
            function, span_boundaries, centre_boundaries, self.position_samples)

    def length_scale(self, function=None, bandwidth_mode='silverman_rule', samples=None, kde_samples=1000):
        # Samples from the estimated pde
        self.kde_samples = kde_samples

        # Initialise the sample positions and their fitness
        if ((self.position_samples is None) | (self.fitness_values is None)) & (function is not None):
            self.initialise(function, samples)

        # Determine the length scale
        indices_1 = np.random.permutation(len(self.fitness_values))
        indices_2 = np.array([*indices_1[1:], indices_1[0]])

        length_scale = (np.abs(self.fitness_values[indices_1] - self.fitness_values[indices_2]) / np.linalg.norm(
            self.position_samples[indices_1, :] - self.position_samples[indices_2, :], axis=1)).reshape(-1, 1)

        # Estimate the bandwidth
        if not isinstance(bandwidth_mode, str):
            self.bandwidth = bandwidth_mode
        else:
            if bandwidth_mode == 'exhaustive':
                order = int(np.ceil(np.log10(np.std(length_scale))))
                coarse_grid = GridSearchCV(KernelDensity(),
                                           {'bandwidth': np.logspace(order / 2 - 3, order / 2 + 3, 25)}, cv=3)
                first_approach = coarse_grid.fit(length_scale).best_estimator_.bandwidth
                fine_grid = GridSearchCV(KernelDensity(), {'bandwidth':
                    np.linspace(0.5 * first_approach, 2 * first_approach, 50)}, cv=3)
                self.bandwidth = fine_grid.fit(length_scale).best_estimator_.bandwidth
            elif bandwidth_mode == 'scott_rule':
                self.bandwidth = 1.06 * np.std(length_scale) * np.power(self.num_samples, -1/5)
            elif bandwidth_mode == 'silverman_rule':
                self.bandwidth = 0.9 * np.min([np.std(length_scale), st.iqr(length_scale)/1.34]) * \
                                 np.power(self.num_samples, -1/5)
            else:
                self.bandwidth = None

        # Estimate the distribution function
        pdf_xvalues = np.linspace(0.9 * length_scale.min(), 1.1 * length_scale.max(), self.kde_samples).reshape(-1, 1)
        pdf_fvalues = np.exp(KernelDensity(bandwidth=self.bandwidth).fit(length_scale).score_samples(pdf_xvalues))

        # Get statistics from raw length_scale values
        dst = st.describe(length_scale)

        # Determine the entropy metric
        entropy_value = (pdf_xvalues[1] - pdf_xvalues[0]) * st.entropy(pdf_fvalues, base=2)

        # Return a dictionary with all the information
        return dict(nob=dst.nobs,
                    raw=length_scale,
                    Min=dst.minmax[0],
                    Max=dst.minmax[1],
                    Avg=dst.mean,
                    Std=np.std(length_scale),
                    Skw=dst.skewness,
                    Kur=dst.kurtosis,
                    IQR=st.iqr(length_scale),
                    Med=np.median(length_scale),
                    MAD=st.median_absolute_deviation(length_scale),
                    KDE_bw=self.bandwidth,
                    PDF_fx=pdf_fvalues,
                    PDF_xs=pdf_xvalues,
                    Entropy=entropy_value)

    @staticmethod
    def _evaluate_positions(function, span_boundaries, centre_boundaries, positions):
        return np.array([function.get_function_value(centre_boundaries + position * (span_boundaries / 2.))
                         for position in positions])

    @staticmethod
    def _normalise_vector(vector):
        return vector / np.max([np.linalg.norm(vector), 1e-23])

    def _levy_walk(self, initial_position, num_steps=1000, alpha=0.5, beta=1.0):

        # Initial position and all the positions are normalised between -1 and 1
        if initial_position == 'rand':
            initial_position = np.random.uniform(-1, 1, self.num_dimensions)
        else:
            if not len(initial_position) == self.num_dimensions:
                raise CharacteriserError('Provide a proper initial position')

        # Initialise the output matrix
        positions = [initial_position]

        # Start the loop for all the steps
        while len(positions) <= num_steps + 1:
            # Get the Levy-distributed step and a point in the hyper-sphere surface
            new_position = positions[-1] + st.levy_stable.rvs(
                alpha, beta, size=self.num_dimensions) * self._normalise_vector(
                np.random.randn(self.num_dimensions))

            # Check if this position is within the domain and register it
            if (new_position > -1.0).all() & (new_position < 1.0).all():
                positions.append(new_position)

        return np.array(positions)




class CharacteriserError(Exception):
    """
    Simple CharacteriserError to manage exceptions.
    """
    pass


if __name__ == '__main__':
    import benchmark_func as bf
    import matplotlib.pyplot as plt

    results_all = []

    for problem_str in bf.__all__:
        problem = eval('bf.' + problem_str + '(2)')

        chsr = Characteriser()
        results_all.append(chsr.length_scale(problem, bandwidth_mode='silverman_rule')['Entropy'])

        print('Evaluated ' + problem_str + '...')
        
    plt.semilogy([res + 1 for res in results_all]), plt.show()



    # problem = bf.Sphere(2)
    # problem.set_search_range(-5, 5)
    #
    # chsr = Characteriser()
    # results = chsr.length_scale(problem, bandwidth_mode='exhaustive')
    # plt.hist(results['raw'], density=True, bins=100), plt.plot(results['PDF_xs'], results['PDF_fx']), plt.show()
    #
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
