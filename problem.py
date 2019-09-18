# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:32:27 2019

@author: jkpvsz

https://github.com/keit0222/optimization-evaluation.git
"""
import numpy as np

class Problem():
    def __init__(self, function, dimensions, desired_fitness):
        
        self.dimensions = dimensions
        self.desired_fitness = desired_fitness
        
        if type(function) == str:
            self.function = locals()[function]()
        else:
            self.function = function
            
    def Jong(x):    
        return np.power(x,2).sum()