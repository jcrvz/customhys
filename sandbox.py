#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:35:32 2019

@author: jkpvsz
"""
# Packages
import numpy as np
import dascop as dso
import benchmark_func as bf
import matplotlib.pyplot as plt

#def run():
    # Problem definition
num_dimensions = 2
num_agents = 50
num_iterations = 100

#problem = bf.Sphere(num_dimensions)
problem = bf.Rosenbrock(num_dimensions)
#problem = bf.Ackley(num_dimensions)
#problem = bf.Griewank(num_dimensions)

is_constrained = True

# Find the proble function : objective function to minimise
problem_function = lambda x : problem.get_func_val(x)

# Define the problem domain
boundaries = (problem.max_search_range, problem.min_search_range)

# Create population
#pop = dso.Population(problem_function,boundaries, num_agents, is_constrained)

# test pour lire les paramètres
simple_heuristics = [("spiral_dynamic", dict(radius=0.9, angle=22.5, sigma=0.1), "all"),
                     ("local_random_walk", dict(probability=0.75, scale=1.0), "greedy")]

verbose_option = True

mh = dso.Metaheuristic(problem_function, boundaries, simple_heuristics,
                   is_constrained, num_agents, num_iterations, verbose_option)
mh.run()

mh.show_performance()

# %%
