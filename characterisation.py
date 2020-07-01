# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""

import numpy as np
from sklearn.neighbors import KernelDensity as ksdensity
import scipy.stats as st


class Characteriser():

    def __init__(self):
        # Define the parameters
        self.bandwidth = 1
        self.kde_samples = 1000
        self.num_repetitions = 30
        self.normalised_boundaries = True
        self.num_dimensions = None

        self.sampling_method = 'levy_walk'
        self.levy_walk_steps = 1000
        self.levy_walk_initial = 'rand'
        self.levy_walk_alpha = 0.5
        self.levy_walk_beta = 1.0

    def set_length_scale(self, bandwidth_mode=1, kde_samples=1000):
        # Estimate the bandwidth
        if not isinstance(bandwidth_mode, str):
            self.bandwidth = bandwidth_mode
        else:
            self.bandwidth = None

        # Samples from the estimated pde
        self.kde_samples = kde_samples

    # Main method: Evaluate one feature for one function
    def get_feature(self, function, feature, samples=None):
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
                position_samples = self._levy_walk(self.levy_walk_initial, self.levy_walk_steps,
                                                   self.levy_walk_alpha, self.levy_walk_beta)

        # Evaluate them in the function
        function_values = self._evaluate_positions(function, span_boundaries, centre_boundaries, position_samples)

        # Determine the length scale



    def _length_scale(self, sample_positions, fitness_values):

        # Determine the length scale
        indices_1 = np.random.permutation(len(fitness_values))
        indices_2 = np.array([*indices_1[1:], indices_1[0]])

        length_scale = np.abs(fitness_values[indices_1] - fitness_values[indices_2]) / np.linalg.norm(
            sample_positions[indices_1, :] - sample_positions[indices_2, :], axis=1)

        # Estimate the distribution function
        pdf_xvalues = np.linspace(0, np.max(length_scale) * 1.1, self.kde_samples)
        pdf_fvalues = np.exp(ksdensity(bandwidth=self.bandwidth).fit(length_scale).score_samples(pdf_xvalues))

        # Get statistics from raw length_scale values
        dst = st.describe(length_scale)

        # Determine the entropy metric
        entropy_value = (pdf_xvalues[1] - pdf_xvalues[0]) * st.entropy(pdf_fvalues, base=2)

        # Return a dictionary with all the information
        return dict(nob=dst.nobs,
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
                    Entropy=entropy_value,
                    ), length_scale

    @staticmethod
    def _evaluate_positions(function, span_boundaries, centre_boundaries, positions):
        return [function(centre_boundaries + position * (span_boundaries / 2.)) for position in positions]

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

        return positions




class CharacteriserError(Exception):
    """
    Simple CharacteriserError to manage exceptions.
    """
    pass


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
