#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:35:32 2019

@author: jkpvsz
"""

from operatos import *

def RandomSearch(problemFunction, nbDimensions=2, nbPopulation=25, nbIterations=100):
    # Initialisation phase
    population = uniformInitialisation(nbDimensions, nbPopulation)
    
    # Evaluate current population
    fitness = problemEvaluation(problemFunction,population)
    
    # Find the best agent and its fitness
    bestAgent, bestFitness = findBest(population, fitness)
    
    # TODO initialise historical data
    
    for iteration in range(1,nbIterations+1):
        # Determine candidate population by perturbating current population
        new_population = simpleRandomWalk(population, scale = 0.0001)
        
        # Evaluate this cantidate population
        new_fitness = problemEvaluation(problemFunction,population)
        
        # Update current population and its fitness
        population, fitness = greedySelector(population, fitness, new_population, new_fitness)
        
        # Update the best agent and its fitness
        bestAgent, bestFitness = findBest(population, fitness, bestAgent, bestFitness)
        
        # Print information per iteration
        print(f"[{iteration}]\t-> {bestFitness}")
    