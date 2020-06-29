# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""

import numpy as np
from sklearn.neighbors import KernelDensity as ksdensity
from scipy.stats import levy_stable


class Characteriser():




def normalise_vector(vector):
    return vector / np.max([np.linalg.norm(vector), 1e-23])

def random_levy_walk(initial_position, num_steps, boundaries):

    # Parameters for generating the Levy random values
    alpha_levy = 0.5
    beta_levy = 1

    # Get the number of dimensions and initialise the output matrix
    num_dimensions = np.size(initial_position)
    positions = [initial_position]

    # Start the loop for all the steps
    while len(positions) <= num_steps + 1:
        # Get the Levy-distributed step and a point in the hyper-sphere surface
        new_position = positions[-1] + levy_stable.rvs(alpha_levy, beta_levy, size=num_dimensions) * \
                       normalise_vector(np.random.randn(num_dimensions))

        # Check if this position is within the domain and register it
        if (new_position > boundaries[0, ]).all() & (new_position < boundaries[1, ]).all():
            positions.append(new_position)

    return positions

def evaluate_positions(function, positions):
    return [function(position) for position in positions]


def length_scale(sample_positions, fitness_values, bandwidth_mode=1, num_kernel_samples=1000):


    # Determine the length scale
    indices_1 = np.random.permutation(len(fitness_values))
    indices_2 = np.array([*indices_1[1:], indices_1[0]])

    length_scale = np.abs(fitness_values[indices_1] - fitness_values[indices_2]) / np.linalg.norm(
        sample_positions[indices_1, :] - sample_positions[indices_2, :], axis=1)
    
    # Estimate the bandwidth
    if not isinstance(bandwidth_mode, str):
        bandwidth = bandwidth_mode
    
    # Estimate the distribution function
    pdf_samples = np.linspace(0, np.max(length_scale) * 1.1, num_kernel_samples)
    log_estimated_pdf = ksdensity(bandwidth=bandwidth).fit(length_scale).score_samples(pdf_samples)


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
