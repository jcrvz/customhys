
# coding: utf-8

# import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

__all__ = ['Ackley1', 'Ackley4', 'Alpine1', 'Alpine2', 'Brown',
           'ChungReynolds', 'Csendes', 'Deb1', 'Deb3', 'DixonPrice',
           'EggHolder', 'Ellipsoid', 'Exp', 'Griewank', 'HyperEllipsoid',
           'KTablet', 'Michalewicz', 'Mishra1', 'Mishra11', 'Mishra2',
           'Mishra7', 'Pathological', 'Perm', 'Pinter', 'PowellSum', 'Qing',
           'Quartic', 'Quintic', 'Rana', 'Rastrigin', 'Rosenbrock',
           'RotatedHyperEllipsoid', 'Salomon', 'Sargan', 'SchafferN1',
           'SchafferN2', 'SchafferN4', 'SchafferN6', 'Schubert', 'Periodic',
           'Schubert3', 'Schubert4', 'SchumerSteiglitz', 'Schwefel',
           'Schwefel12', 'Schwefel204', 'Schwefel220', 'Schwefel221',
           'Schwefel222', 'Schwefel223', 'Schwefel225', 'Schwefel226',
           'Sphere', 'Step', 'Step2', 'Step3', 'StepInt', 'StrechedVSineWave',
           'StyblinskiTang', 'SumSquares', 'Trid10', 'Trid6', 'Trigonometric1',
           'Trigonometric2', 'WWavy', 'Weierstrass', 'Whitley', 'XinSheYang1',
           'XinSheYang2', 'XinSheYang3', 'XinSheYang4', 'Zakharov', 'HappyCat']


# %% Basic function class
class OptimalBasic:
    def __init__(self, variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([0]*self.variable_num)
        self.min_search_range = np.array([0]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True}
        self.plot_place = 0.25
        self.func_name = ''
        self.save_dir = os.path.dirname(os.path.abspath(__file__)) +\
            '\\function_plots\\'
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return (self.max_search_range, self.min_search_range)

    def get_func_val(self, variables):
        return -1

    def plot(self, samples=100):
        x = np.linspace(self.min_search_range[0], self.max_search_range[0],
                        samples)
        y = np.linspace(self.min_search_range[1], self.max_search_range[1],
                        samples)
        X, Y = np.meshgrid(x, y)
        Z = []
        for xy_list in zip(X, Y):
            z = []
            for xy_input in zip(xy_list[0], xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[2:self.variable_num]))
                z.append(self.get_func_val(np.array(tmp)))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure(figsize=[3, 2], dpi=333)
        ls = LightSource(azdeg=90, altdeg=45)
        rgb = ls.shade(Z, plt.cm.coolwarm)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, facecolors=rgb)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title(self.func_name)
        plt.tight_layout()
        plt.show()

    def save_fig(self):
        x = np.arange(self.min_search_range[0], self.max_search_range[0],
                      self.plot_place, dtype=np.float32)
        y = np.arange(self.min_search_range[1], self.max_search_range[1],
                      self.plot_place, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        Z = []
        for xy_list in zip(X, Y):
            z = []
            for xy_input in zip(xy_list[0], xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[2:self.variable_num]))
                z.append(self.get_func_val(np.array(tmp)))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X, Y, Z)
        plt.savefig(self.save_dir + self.func_name + '.png')
        plt.close()

# %% Optimization benchmark function group


# 1 - Class Ackley 1 function
class Ackley1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([35]*self.variable_num)
        self.min_search_range = np.array([-35]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.func_name = 'Ackley 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, a=20., b=0.2, c=2.*np.pi):
        return a + np.e - (a * np.exp(-b * np.sqrt(
            np.sum(np.square(variables))/self.variable_num)) +
            np.exp(np.sum(np.cos(c * variables))/self.variable_num))


# 4 - Class Ackley 4 function
class Ackley4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([35]*self.variable_num)
        self.min_search_range = np.array([-35]*self.variable_num)
        self.optimal_solution = np.array([-1.479252, -0.739807])
        self.global_optimum_solution = -3.917275
        self.func_name = 'Ackley 4'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.exp(-0.2) * np.sum(np.sqrt(
            np.square(variables[:-1]) + np.square(variables[1:]))) + \
            3. * np.sum(np.cos(2. * variables[:-1]) +
                        np.sin(2. * variables[:-1]))


# 6 - Class Alpine 1 function
class Alpine1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10]*self.variable_num)
        self.min_search_range = np.array([-10]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.func_name = 'Alpine 1'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.abs(variables * np.sin(variables) + 0.1 * variables))


# 7 - Class Alpine 2 function
class Alpine2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10]*self.variable_num)
        self.min_search_range = np.array([0]*self.variable_num)
        self.optimal_solution = np.array([7.917]*self.variable_num)
        self.global_optimum_solution = 2.808 ** self.variable_num
        self.func_name = 'Alpine 2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.prod(np.sqrt(variables) * np.sin(variables))


# 25 - Class Brown function
class Brown(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([4]*self.variable_num)
        self.min_search_range = np.array([-1]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.func_name = 'Brown'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': True}

    def get_func_val(self, variables):
        xi = np.square(variables[:-1])
        xi1 = np.square(variables[1:])
        return np.sum(np.power(xi, xi1 + 1) + np.power(xi1, xi + 1))


# 34 - Class Chung Reynolds function
class ChungReynolds(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100]*self.variable_num)
        self.min_search_range = np.array([-100]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.func_name = 'Chung Reynolds'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.square(np.sum(np.square(variables)))


# 40 - Class Csendes function
class Csendes(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1]*self.variable_num)
        self.min_search_range = np.array([-1]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.func_name = 'Csendes'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.square(np.sum(np.square(variables)))


# 43 - Class Deb 1 function
class Deb1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1]*self.variable_num)
        self.min_search_range = np.array([-1]*self.variable_num)
        self.optimal_solution = np.array([-0.1]*self.variable_num)
        self.global_optimum_solution = -1
        self.func_name = 'Deb 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.power(np.sin(5. * np.pi * variables), 6.)) / (
            -self.variable_num)


# 44 - Class Deb 3 function
class Deb3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1]*self.variable_num)
        self.min_search_range = np.array([0]*self.variable_num)
        self.optimal_solution = np.array([np.power(1/10 + 0.05, 4/3)] *
                                         self.variable_num)
        self.global_optimum_solution = -1
        self.func_name = 'Deb 3'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.power(np.sin(5. * np.pi * (
            np.power(variables, 3/4) - 0.05)), 6.)) / (-self.variable_num)


# 48 - Class Dixon & Price function
class DixonPrice(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10]*self.variable_num)
        self.min_search_range = np.array([-10]*self.variable_num)
        self.optimal_solution = np.power(2., np.power(
            2., - np.arange(self.variable_num)) - 1)
        self.global_optimum_solution = 0
        self.func_name = 'Dixon-Price'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.square(variables[0] - 1) + np.sum([
            i * np.square(2 * np.square(variables[i]) - variables[i - 1])
            for i in range(1, self.variable_num)])


# 53 - Class Egg Holder function
class EggHolder(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([512.]*self.variable_num)
        self.min_search_range = np.array([-512.]*self.variable_num)
        self.optimal_solution = np.array([512., 404.2319])
        self.global_optimum_solution = -959.6407
        self.plot_place = 5
        self.func_name = 'Egg Holder'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': False}

    def get_func_val(self, variables):
        xi = variables[:-1]
        xi1 = variables[1:]
        return np.sum(-(xi1 + 47.) * np.sin(np.sqrt(
            np.abs(xi1 + xi/2. + 47.))) - xi * np.sin(np.sqrt(np.abs(
                xi - xi1 - 47.))))


# 54 - Class Exponential function
class Exp(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.]*self.variable_num)
        self.min_search_range = np.array([-1.]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 1
        self.plot_place = 5
        self.func_name = 'Exponential'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': True}

    def get_func_val(self, variables):
        return -np.exp(-0.5 * np.sum(np.square(variables)))


# 59 - Class Griewank function
class Griewank(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Griewank'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.square(variables))/4000. - np.prod(np.cos(
            variables / np.sqrt(np.arange(self.variable_num) + 1))) + 1

# Class Happy Cat function
class HappyCat(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([-1.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Happy Cat'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, alpha=1/8):
        x1 = np.sum(variables)
        x2 = np.sum(np.square(variables))
        return np.power(x2 - self.variable_num, 2. * alpha) + (x2/2 + x1) / (
            self.variable_num) + 1/2


# 74 - Class Mishra 1 function
class Mishra1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.]*self.variable_num)
        self.min_search_range = np.array([0.]*self.variable_num)
        self.optimal_solution = np.array([1.]*self.variable_num)
        self.global_optimum_solution = 2.
        self.plot_place = 10.
        self.func_name = 'Mishra 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        g_funct = self.variable_num - np.sum(variables[:self.variable_num-1])
        return np.power(1 + g_funct, g_funct)


# 75 - Class Mishra 2 function
class Mishra2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.]*self.variable_num)
        self.min_search_range = np.array([0.]*self.variable_num)
        self.optimal_solution = np.array([1.]*self.variable_num)
        self.global_optimum_solution = 2.
        self.plot_place = 10.
        self.func_name = 'Mishra 2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        g_funct = self.variable_num - 0.5 * np.sum(
            variables[:self.variable_num-1] + variables[1:self.variable_num])
        return np.power(1 + g_funct, g_funct)


# 80 - Class Mishra 7 function
class Mishra7(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([np.power(np.math.factorial(
            self.variable_num), 1/self.variable_num)] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Mishra 7'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.square(np.prod(variables) - np.math.factorial(
            self.variable_num))


# 84 - Class Mishra 11 function
class Mishra11(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([0.]*self.variable_num)
        self.optimal_solution = np.array([np.nan] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Mishra 11'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.square(np.sum(np.abs(variables))/self.variable_num -
                         np.power(np.prod(np.abs(variables)),
                                  1/self.variable_num))


# 87 - Class Pathological function
class Pathological(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Pathological'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        xi = variables[:-1]
        xj = variables[1:]
        return np.sum(0.5 + np.square(np.sin(np.sqrt(
            100 * np.square(xi) + np.square(xj))) - 0.5) / (
                1 + 0.001 * np.power(xi - xj, 4)))


# 89 - Class Pinter function
class Pinter(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Pinter'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        A = variables[:-2] * np.sin(variables[1:-1]) + np.sin(variables[2:])
        B = np.square(variables[:-2]) - 2 * variables[1:-1] +\
            3 * variables[2:] - np.cos(variables[1:-1]) + 1
        i = np.arange(self.variable_num) + 1
        return np.sum(i * np.square(variables)) + np.sum(
            20 * i[1:-1] * np.square(np.sin(A))) + np.sum(
                i[1:-1] * np.log10(1 + i[1:-1] * np.square(B)))


# Class Periodic function
class Periodic(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.9
        self.plot_place = 10.
        self.func_name = 'Periodic'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return 1 + np.sum(np.square(np.sin(variables))) - 0.1 * np.exp(
            np.sum(np.square(variables)))


# 93 - Class Powell Sum function
class PowellSum(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.]*self.variable_num)
        self.min_search_range = np.array([-1.]*self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Powel Sum'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.power(np.abs(variables), np.arange(
            self.variable_num) + 2))


# 98 - Class Qing function
class Qing(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.]*self.variable_num)
        self.min_search_range = np.array([-500.]*self.variable_num)
        self.optimal_solution = np.array(np.sqrt(np.arange(
            self.variable_num) + 1))
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Powel Sum'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.square(np.square(variables) - (np.arange(
            self.variable_num) + 1)))


# 100 - Class Quartic function
class Quartic(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.28]*self.variable_num)
        self.min_search_range = np.array([-1.28]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Quartic'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.power(variables, 4) * (np.arange(
            self.variable_num) + 1))


# 101 - Class Quintic function
class Quintic(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([-1]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Quintic'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return np.sum(np.abs(np.power(variables, 5) - 3 * np.power(
            variables, 4) + 4 * np.power(variables, 3) + 2 * np.power(
                variables, 2) - 10 * variables - 4))


# 102 - Class Rana function
class Rana(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.000001]*self.variable_num)
        self.min_search_range = np.array([-500.000001]*self.variable_num)
        self.optimal_solution = np.array([-500.]*self.variable_num)
        self.global_optimum_solution = -511.70430*self.variable_num+511.68714
        self.plot_place = 10.
        self.func_name = 'Rana'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        t1 = np.sqrt(np.abs(variables[1:] + variables[:-1] + 1))
        t2 = np.sqrt(np.abs(variables[1:] - variables[:-1] + 1))
        return np.sum((variables[1:] + 1) * np.cos(t1) * np.sin(t1) +
                      variables[:-1] * np.cos(t1) * np.sin(t2))


# Class Rastrigin function
class Rastrigin(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Rastrigin'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables):
        return 10. * self.variable_num + np.sum(
            np.square(variables) - 10. * np.cos(2. * np.pi * variables))


# Class Ridge function
class Ridge(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.]*self.variable_num)
        self.min_search_range = np.array([-5.]*self.variable_num)
        self.optimal_solution = np.array([self.min_search_range[0],
                                          *[0.]*(self.variable_num - 1)])
        self.global_optimum_solution = 0.
        self.func_name = 'Ridge'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, d=2., alpha=0.1):
        return variables[0] + d * np.power(np.sum(np.square(variables)), alpha)


# 105 - Class Rosenbrock function
class Rosenbrock(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([30.]*self.variable_num)
        self.min_search_range = np.array([-30.]*self.variable_num)
        self.optimal_solution = np.array([1.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Rosenbrock'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(100. * np.square(variables[1:] - np.square(
            variables[:-1])) + np.square(variables[:-1] - 1))


# 110 - Class Salomon function
class Salomon(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Salomon'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return 1 - np.cos(2 * np.pi * np.sqrt(np.sum(np.square(variables)))) +\
            0.1 * np.sqrt(np.sum(np.square(variables)))


# 111 - Class Sargan function
class Sargan(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Sargan'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return self.variable_num * np.sum(np.square(variables) + 0.4 * np.sum(
            np.multiply(1 - np.identity(self.variable_num), np.outer(
                variables, variables)), 0))


# 117 - Class Schumer-Steiglitz function
class SchumerSteiglitz(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schumer-Steiglitz'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.power(variables, 4))


# 118 - Class Schwefel function
class Schwefel(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables, alpha=0.5):
        return np.power(np.sum(np.square(variables)), alpha)


# 119 - Class Schwefel 1.2 function
class Schwefel12(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel 1.2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.square(np.sum(np.tril(np.outer(
            np.ones(self.variable_num), variables)), 1)))


# 120 - Class Schwefel 2.04 function
class Schwefel204(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([0.]*self.variable_num)
        self.optimal_solution = np.array([1.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel 2.04'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return np.sum(np.square(variables - 1) + np.square(
            variables[0] - np.square(variables)))


# 122 - Class Schwefel 2.20 function
class Schwefel220(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel 2.20'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return -np.sum(np.abs(variables))


# 123 - Class Schwefel 2.21 function
class Schwefel221(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel 2.21'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.max(np.abs(variables))


# 124 - Class Schwefel 2.22 function
class Schwefel222(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel 2.22'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.abs(variables)) + np.prod(np.abs(variables))


# 125 - Class Schwefel 2.23 function
class Schwefel223(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel 2.23'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.power(variables, 10))


# 127 - Class Schwefel 2.25 function
class Schwefel225(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([0.]*self.variable_num)
        self.optimal_solution = np.array([1.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schwefel 2.25'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return np.sum(np.square(variables[1:] - 1) + np.square(
            variables[0] - np.square(variables[1:])))


# 128 - Class Schwefel 2.26 function
class Schwefel226(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.]*self.variable_num)
        self.min_search_range = np.array([-500.]*self.variable_num)
        self.optimal_solution = np.array([np.square(np.pi * 1.5)] *
                                         self.variable_num)
        self.global_optimum_solution = -418.983
        self.plot_place = 0.25
        self.func_name = 'Schwefel 2.26'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return np.sum(variables * np.sin(np.abs(variables))) / (
            -self.variable_num)


# 133 - Class Schubert function
class Schubert(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([np.nan]*self.variable_num)
        self.global_optimum_solution = -186.7309
        self.plot_place = 0.25
        self.func_name = 'Schubert'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        j = np.arange(1, 6)
        return np.prod(np.sum(np.cos(np.outer(variables, j + 1) + np.outer(
            np.ones(self.variable_num), j)), 1))


# 134 - Class Schubert 3 function
class Schubert3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([np.nan]*self.variable_num)
        self.global_optimum_solution = -29.6733337
        self.plot_place = 0.25
        self.func_name = 'Schubert 3'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        j = np.arange(1, 6)
        return np.sum(np.sum(np.outer(np.ones(self.variable_num), j) *
                             np.sin(np.outer(variables, j + 1) + np.outer(
                                 np.ones(self.variable_num), j)), 1))


# 135 - Class Schubert 4 function
class Schubert4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([np.nan]*self.variable_num)
        self.global_optimum_solution = -25.740858
        self.plot_place = 0.25
        self.func_name = 'Schubert 4'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        j = np.arange(1, 6)
        return np.sum(np.sum(np.outer(np.ones(self.variable_num), j) *
                             np.cos(np.outer(variables, j + 1) + np.outer(
                                 np.ones(self.variable_num), j)), 1))


# 136 - Class Schaffer N6 function
class SchafferN6(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schaffer N6'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return 0.5 * (self.variable_num - 1) + np.sum((np.square(
            np.sin(np.sqrt(np.square(variables[:-1]) + np.square(
                variables[1:])))) - 0.5) / np.square(1 + 0.001 * (
                    variables[:-1] + variables[1:])))


# 136* - Class Schaffer N1 function
class SchafferN1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schaffer N1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return 0.5 * (self.variable_num - 1) + np.sum((np.square(
            np.sin(np.square(np.square(variables[:-1]) + np.square(
                variables[1:])))) - 0.5) / np.square(1 + 0.001 * (
                    variables[:-1] + variables[1:])))


# 136* - Class Schaffer N2 function
class SchafferN2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schaffer N2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return 0.5 * (self.variable_num - 1) + np.sum((np.square(np.sin(
            np.square(variables[:-1]) + np.square(variables[1:]))) - 0.5) /
            np.square(1 + 0.001 * (variables[:-1] + variables[1:])))


# 136* - Class Schaffer N4 function
class SchafferN4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 0.25
        self.func_name = 'Schaffer N4'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return 0.5 * (self.variable_num - 1) + np.sum((np.square(np.sin(
            np.abs(np.square(variables[:-1]) + np.square(
                variables[1:])))) - 0.5) / np.square(1 + 0.001 * (
                    variables[:-1] + variables[1:])))


# 137 - Class Sphere function
class Sphere(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Sphere'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return np.sum(np.square(variables))


# 138 - Class Step function
class Step(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Step'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.floor(np.abs(variables)))


# 139 - Class Step 2 function
class Step2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.5]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Step 2'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.square(np.floor(variables + 0.5)))


# 140 - Class Step 3 function
class Step3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Step 3'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.floor(np.square(variables)))


# 141 - Class Step Int function
class StepInt(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 25 - 6 * self.variable_num
        self.plot_place = 10
        self.func_name = 'Step Int'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.floor(variables)) + 25


# 142 - Class Streched V Sine Wave function
class StrechedVSineWave(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Streched V Sine Wave'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Multimodal': False}

    def get_func_val(self, variables):
        f10 = np.power(np.square(variables[:-1]) +
                       np.square(variables[1:]), 0.10)
        f25 = np.power(np.square(variables[:-1]) +
                       np.square(variables[1:]), 0.25)
        return np.sum(f25 * (np.square(np.sin(f10)) + 0.1))


# 143 - Class Sum Squares function
class SumSquares(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Sum Squares'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.arange(1, self.variable_num + 1)*np.square(variables))


# 150 - Class Trid 6 function
class Trid6(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.square(self.variable_num)] *
                                         self.variable_num)
        self.min_search_range = np.array([-np.square(self.variable_num)] *
                                         self.variable_num)
        self.optimal_solution = np.array(np.arange(1, self.variable_num + 1) *
                                         (self.variable_num + 1 - np.arange(
                                             1, self.variable_num + 1)))
        self.global_optimum_solution = -self.variable_num * (
            self.variable_num + 4) * (self.variable_num - 1)/6
        self.plot_place = 10
        self.func_name = 'Trid 6'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return np.sum(np.square(variables - 1)) - np.sum(
            variables[1:] * variables[:-1])


# 151 - Class Trid 10 function
class Trid10(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array(np.arange(1, self.variable_num + 1) *
                                         (self.variable_num + 1 - np.arange(
                                             1, self.variable_num + 1)))
        self.global_optimum_solution = -200.
        self.plot_place = 10
        self.func_name = 'Trid 10'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return np.sum(np.square(variables - 1)) - np.sum(
            variables[1:] * variables[:-1])


# 153 - Class Trigonometric 1 function
class Trigonometric1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Trigonometric 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        x = np.outer(variables, np.ones(self.variable_num))
        i = np.outer(np.arange(1, self.variable_num + 1),
                     np.ones(self.variable_num))
        return np.sum(np.square(self.variable_num - np.sum(np.cos(x) + i * (
            1 - np.cos(x.T) - np.sin(x.T)), 0)))


# 154 - Class Trigonometric 2 function
class Trigonometric2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([0.9] * self.variable_num)
        self.global_optimum_solution = 1.
        self.plot_place = 10
        self.func_name = 'Trigonometric 2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return 1 + np.sum(8. * np.square(np.sin(7. * np.square(
            variables - 0.9))) + 6. * np.square(np.sin(14. * np.square(
                variables - 0.9))) + np.square(variables - 0.9))


# 165 - Class W / Wavy function
class WWavy(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi] * self.variable_num)
        self.min_search_range = np.array([-np.pi] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'W / Wavy'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables, k=10):
        return 1 - np.sum(np.cos(k * variables) * np.exp(-0.5 * np.exp(
            -np.square(variables)))) / self.variable_num


# 166 - Class Weierstrass function
class Weierstrass(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([0.5] * self.variable_num)
        self.min_search_range = np.array([-0.5] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Weierstrass'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables, kmax=20, a=0.5, b=3.):
        x = np.outer(variables + 0.5, np.ones(kmax + 1))
        k = np.outer(np.ones(self.variable_num), np.arange(kmax + 1))
        return np.sum(np.sum(np.power(a, k) * np.cos(
            2. * np.pi * np.power(b, k) * x), 1) - self.variable_num * np.sum(
                np.power(a, k) * np.cos(np.pi * np.power(b, k)), 1))


# 167 - Class Whitley function
class Whitley(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.24] * self.variable_num)
        self.min_search_range = np.array([-10.24] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10
        self.func_name = 'Whitley'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        x = np.outer(variables, np.ones(self.variable_num))
        X = 100. * np.square(np.square(x) - x.T) + np.square(1 - x.T)
        return np.sum(np.square(X) / 4000. - np.cos(X + 1))


# 169 - Class Xin-She Yang 1 function
class XinSheYang1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.]*self.variable_num)
        self.min_search_range = np.array([-5.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Xin-She Yang 1'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.random.rand(self.variable_num) *
                      np.power(np.abs(variables), np.arange(
                          1, self.variable_num + 1)))


# 170 - Class Xin-She Yang 2 function
class XinSheYang2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([2.*np.pi]*self.variable_num)
        self.min_search_range = np.array([-2.*np.pi]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Xin-She Yang 2'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.abs(variables)) * np.exp(-np.sum(
            np.sin(np.square(variables))))


# 171 - Class Xin-She Yang 3 function
class XinSheYang3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([20.]*self.variable_num)
        self.min_search_range = np.array([-20.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Xin-She Yang 3'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables, m=5, beta=15):
        return np.exp(-np.sum(np.power(variables/beta, 2. * m))) - \
                      2. * np.exp(-np.sum(np.square(variables))) * \
                      np.prod(np.square(np.cos(variables)))


# 172 - Class Xin-She Yang 4 function
class XinSheYang4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Xin-She Yang 4'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return (np.sum(np.square(np.sin(variables))) - np.exp(-np.sum(
            np.square(variables)))) * np.exp(-np.sum(np.square(
                np.sin(np.abs(variables)))))


# 173 - Class Zakharov function
class Zakharov(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-5.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Zakharov'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        ixi = np.arange(1, self.variable_num) * variables
        return np.sum(np.square(variables)) + np.square(0.5 * np.sum(ixi)) + \
            np.power(0.5 * np.sum(ixi), 4)


# Class Styblinski-Tang function
class StyblinskiTang(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.]*self.variable_num)
        self.min_search_range = np.array([-5.]*self.variable_num)
        self.optimal_solution = np.array([-2.903534]*self.variable_num)
        self.global_optimum_solution = -39.16599*self.variable_num
        self.func_name = 'Styblinski-Tang'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables):
        return 0.5 * np.sum(np.power(variables, 4) - 16. *
                            np.square(variables) + 5. * variables)


# Class Ellipsoid function
class Ellipsoid(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Ellipsoid'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.square(np.power(10, np.arange(
            self.variable_num) / (self.variable_num - 1)) * variables))


# Class Hyper-Ellipsoid function
class HyperEllipsoid(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Hyper-Ellipsoid'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.arange(1, self.variable_num + 1) *
                      np.square(variables))


# Class Rotated-Hyper-Ellipsoid function
class RotatedHyperEllipsoid(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([65.536]*self.variable_num)
        self.min_search_range = np.array([-65.536]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Rotated-Hyper-Ellipsoid'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        return np.sum(np.tril(np.outer(np.ones(self.variable_num),
                                       np.square(variables))))


# Class Michalewicz function
class Michalewicz(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi]*self.variable_num)
        self.min_search_range = np.array([0.]*self.variable_num)
        self.optimal_solution = np.array([2.20319, 1.57049])
        self.global_optimum_solution = -1.8013  # In case of variable_num == 2
        self.plot_place = 0.1
        self.func_name = 'Michalewicz'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': True}

    def get_func_val(self, variables, m=10.):
        return -np.sum(np.sin(variables) * np.power(np.sin(
            np.arange(1, self.variable_num + 1) * np.square(variables)/np.pi),
            2*m))


# Class K-Tablet function
class KTablet(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'K-Tablet'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables):
        k = int(self.variable_num/4)
        return np.sum(variables[:k]) + np.sum(np.square(100. * variables[k:]))


# Class Perm function
class Perm(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.]*self.variable_num)
        self.min_search_range = np.array([-1.]*self.variable_num)
        self.optimal_solution = 1/np.arange(1, self.variable_num + 1)
        self.global_optimum_solution = 0.
        self.plot_place = 0.1
        self.func_name = 'Perm'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Multimodal': False}

    def get_func_val(self, variables, beta=1.):
        x = np.outer(variables, np.ones(self.variable_num))
        j = np.outer(np.ones(self.variable_num), np.arange(
            self.variable_num) + 1)
        return np.sum(np.square(np.sum((j.T + beta) * (
            np.power(x, j) - np.power(1/j.T, j)), 0)))


# %%
if __name__ == '__main__':
    q = Sphere(2)
    q.plot()
