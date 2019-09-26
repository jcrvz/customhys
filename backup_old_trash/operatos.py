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
import random

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

# 0.1. Rescale population from [lower,upper] to [-1,1] per dimension
# TODO write this function

# 0.2. Rescale population from [-1,1] to [lower,upper] per dimension
def rescaleBack(agent, lowerBoundaries, upperBoundaries):
    if agent.shape[0] == 1:
        agent.transpose()
    # y = ((b - a) * x + (a + b)) / 2
    return ((upperBoundaries + lowerBoundaries) + agent.transpose() * (upperBoundaries - lowerBoundaries)) / 2

# 0.2. Evaluate problem function for each agent into population
def problemEvaluation(problemFunction,population):
    # TODO rescale agents
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
    population = np.full((nbDimensions, nbPopulation), np.nan)
    for agent in range(0, nbPopulation):
        for dimension in range(0, nbDimensions):
            population[dimension, agent] = random.uniform(-1,1)        
    return population

#TODO Add more initialisation operators like grid, boundary, etc.

# ---------------------------------------------------------------------------
# 2. Perturbators
# ---------------------------------------------------------------------------

# 2.1. 
def simpleRandomWalk(population, scale = 0.01):
    #TODO if nb arguments are not provided, then calculate them   
    #new_population = np.full(population.shape, np.nan)
    for agent in range(0, population.shape[1]):
        for dimension in range(0, population.shape[0]):
            population[dimension, agent] += scale * random.uniform(-1,1)        
    return population

# ---------------------------------------------------------------------------
# 3. Selectors
# ---------------------------------------------------------------------------

# 3.1. 
def findBest(population, fitness, bestAgent=None, bestFitness=None):
    current_bestAgent = population[:,fitness.argmin()]
    current_bestFitness = fitness.min()
    if bestFitness:
        if current_bestFitness > bestFitness:
            current_bestAgent = bestAgent
            current_bestFitness = bestFitness                
    return current_bestAgent, current_bestFitness

# 3.2. 
def greedySelector(population, fitness, new_population, new_fitness):
    # TODO improve by using np.ndarray.min(,axis) and np.ndarray.argmin(,axis)
    for agent in range(0,population.shape[1]):
        if new_fitness[agent] <= fitness[agent]:
            fitness[agent] = new_fitness[agent]
            population[:,agent] = new_population[:,agent]
    return population, fitness
    