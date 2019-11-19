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

# Find the problem function : objective function to minimise
problem_function = lambda x : problem.get_func_val(x)

# Define the problem domain
boundaries = (problem.min_search_range, problem.max_search_range)

# Create population
#pop = dso.Population(problem_function,boundaries, num_agents, is_constrained)

# test pour lire les paramètres
#simple_heuristics = [("spiral_dynamic", dict(radius=0.9, angle=22.5, sigma=0.1), "all"),
#                     ("local_random_walk", dict(probability=0.75, scale=1.0), "greedy")]

# Pairing mechanics
# evenodd, additional parameters = none
# rank, additional parameters = none
# cost, additional parameters = none
# tournament, additional parameters = (tournament_size = 2, probability = 1.0)

# Crossover mechanics
# single, aditional parameters = none
# two, aditional parameters = none
# uniform, aditional parameters = none
# blend, aditional parameters = none
# linear, aditional parameters = coefficients[0.5, 0.5] -> coef_1 * father + coef_2 * mother

# Mutation mechanichs
# uniform, additional parameters = sigma (step scale)
# normal, additional parameters = sigma (standard deviation)

# Genetic algorithm
simple_heuristics = [
        ("ga_crossover", dict(pairing="tournament", crossover="uniform", mating_pool_factor=0.1, 
                          coefficients=[0.5,0.5]), "all"),
    ("ga_mutation", dict(elite_rate=0.1, mutation_rate=0.25, 
                                          distribution="normal", sigma=1.0), "all")]

verbose_option = True

mh = dso.Metaheuristic(problem_function, boundaries, simple_heuristics, is_constrained, num_agents, num_iterations,
                       verbose_option)

# to comment
mh.pop.initialise_uniformly()
mh.pop.evaluate_fitness()

mh.run()

mh.show_performance()

# %%
"""
par1 = np.array([10,20,30])
par2 = np.array([0.1,0.2,0.3,0.4,0.5])
par3 = np.array([1,2,3,4,5])
data = np.meshgrid(par1,par2,par3)
comb = np.full((3,par1.size * par2.size * par2.size),np.nan)
cc = np.array([data[x].flatten() for x in range(len(data))])
"""

