#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains several population-based search operators for continuous optimisation problems

* VARIABLES:
    + nbDimensions: number of dimensions (problem domain)
    + nbPopulation: number of agents (population size)
    
* OPERATORS:
    + INITIALISATION OPERATORS:
        * uniformInitialisation: Initialise population using a random uniform
                                 distribution in [-1,1]

Created on Sat Sep 14 16:33:14 2019

@author: jkpvsz
"""

import numpy as np

# -- to delete
def test():
    x = uniformInitialisation(2,3)
    y = simpleRandomWalk(x,1)
    fx = problemEvaluation(Jong,x)
    fy = problemEvaluation(Jong,y)
    
    z, fz =greedySelector(x,fx,y,fy)
    
    print(f"x = {x}, \nfx = {fx}\n")
    print(f"y = {y}, \nfy = {fy}\n")
    print(f"z = {z}, \nfz = {fz}")

def Jong(x):
    return np.power(x,2).sum()
# -- to delete

# ---------------------------------------------------------------------------
# 0. Basics
# ---------------------------------------------------------------------------

# 0.1. 
def problemEvaluation(problemFunction,population):
    # Initialise populationFitness array with elements equal to nan
    fitness = np.full(population.shape[1],np.nan)
    for agent in range(0,population.shape[1]):
        fitness[agent] = problemFunction(population[:,agent])    
    return fitness

# ---------------------------------------------------------------------------
# 1. Initialisators
# ---------------------------------------------------------------------------

# 1.1. Initialise population using a random uniform distribution in [-1,1]]
def uniformInitialisation(nbDimensions, nbPopulation):
    # Require:  nbDimensions, nbPopulation : int
    # Ensure:   out: ndarray, shape (nbDimensions, nbPopulation), 
    #                elements with uniform distribution between -1 and 1
    return 2 * np.random.rand(nbDimensions,nbPopulation) - 1

#TODO Add more initialisation operators like grid, boundary, etc.

# ---------------------------------------------------------------------------
# 2. Perturbators
# ---------------------------------------------------------------------------

# 2.1. 
def simpleRandomWalk(population, scale = 0.01):
    #TODO if nb arguments are not provided, then calculate them    
    return population + scale *  (2 * np.random.rand(population.shape[0], population.shape[1]) - 1)# nbDimensions, nbPopulation

# ---------------------------------------------------------------------------
# 3. Selectors
# ---------------------------------------------------------------------------

# 3.1. 
def greedySelector(population,fitness,new_population,new_fitness):
    for agent in range(0,population.shape[1]):
        if new_fitness[agent] <= fitness[agent]:
            fitness[agent] = new_fitness[agent]
            population[:,agent] = new_population[:,agent]
    return population, fitness
    