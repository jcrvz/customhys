# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:35:32 2020

@author: jcrvz
"""


import benchmark_func as bf

all_functions = bf.__all__

for ii in range(len(all_functions)):
    function_name = all_functions[ii]
    print('Plotting... Function {} of {}: {}'.format(
        ii+1, len(all_functions), function_name))
    func = eval('bf.' + function_name + '(2)')
    func.plot(samples=125, resolution=300)
    func.save_fig()
    bf.plt.close('all')