# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""

import numpy as np
from sklearn.neighbors import KernelDensity as ksdensity
from scipy.stats import levy_stable

def random_walk():
    r = levy_stable.rvs(alpha, beta, size=1000)


def length_scale(sample_positions, fitness_values, bandwidth_mode=1, num_kernel_samples=1000):


    # Determine the length scale
    indices_1 = np.random.permutation(len(fitness_values))
    indices_2 = np.array([*indices_1[1:], indices_1[0]])

    length_scale = np.abs(fitness_values[indices_1] - fitness_values[indices_2]) / np.linalg.norm(
        sample_positions[indices_1, :] - sample_positions[indices_2, :], axis=1)
    
    # Estimate the bandwidth
    if not isinstance(bandwidth_mode, 'str'):
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
