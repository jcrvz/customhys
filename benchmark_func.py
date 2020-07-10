# -*- coding: utf-8 -*-
"""
This model is an upgraded version of the Keita Tomochika's module in
https://github.com/keit0222/optimization-evaluation. The current module only contain N-dimensional functions. These
functions are listed in ``__all__``.

All these functions are based and revisited on the following research papers and web sites:

- Momin Jamil and Xin-She Yang, A literature survey of benchmark functions for global optimization problems,
Int. Journal of Mathematical Modelling and Numerical Optimisation, Vol. 4, No. 2, pp. 150–194 (2013), arXiv:1308.4008
- Mazhar Ansari Ardeh, https://github.com/mazhar-ansari-ardeh, http://benchmarkfcns.xyz
- Sonja Surjanovic and Derek Bingham, Simon Fraser University, https://www.sfu.ca/~ssurjano/optimization.html
- Ali R. Al-Roomi (2015). Unconstrained Single-Objective Benchmark Functions Repository. Halifax, Nova Scotia,
Canada, Dalhousie University, Electrical and Computer Engineering. https://www.al-roomi.org/benchmarks/unconstrained
- B.Y. Qu, J.J. Liang, Z.Y. Wang, Q. Chen, P.N. Suganthan. Novel benchmark functions for continuous multimodal
optimization with comparative results. Swarm and Evolutionary Computation, 26 (2016), 23-34.

Created on Tue Sep 17 14:29:43 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.colors import LightSource
from matplotlib import rcParams

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)

__all__ = ['Ackley1', 'Ackley4', 'Alpine1', 'Alpine2', 'Bohachevsky', 'Brent', 'Brown', 'CarromTable', 'ChungReynolds',
           'Cigar', 'CosineMixture', 'CrossInTray', 'CrossLegTable', 'CrownedCross', 'Csendes', 'Deb1', 'Deb2',
           'DeflectedCorrugatedSpring', 'DixonPrice', 'DropWave', 'EggHolder', 'Ellipsoid', 'ExpandedDecreasingMinima',
           'ExpandedEqualMinima', 'ExpandedFiveUnevenPeakTrap', 'ExpandedTwoPeakTrap', 'ExpandedUnevenMinima',
           'Exponential', 'F2', 'Giunta', 'Griewank', 'HappyCat', 'HyperEllipsoid', 'InvertedCosineWave',
           'JennrichSampson', 'KTablet', 'Katsuura', 'Levy', 'LunacekN01', 'LunacekN02', 'Michalewicz', 'Mishra1',
           'Mishra2', 'Mishra7', 'Mishra11', 'ModifiedVincent', 'NeedleEye', 'Pathological', 'Periodic', 'Perm01',
           'Perm02', 'Pinter', 'PowellSum', 'Price01', 'Qing', 'Quartic', 'Quintic', 'Rana', 'Rastrigin', 'Ridge',
           'Rosenbrock', 'RotatedHyperEllipsoid', 'Salomon', 'Sargan', 'SchafferN1', 'SchafferN2', 'SchafferN3',
           'SchafferN4', 'SchafferN6', 'Schubert', 'Schubert3', 'Schubert4', 'SchumerSteiglitz', 'Schwefel',
           'Schwefel12', 'Schwefel204', 'Schwefel220', 'Schwefel221', 'Schwefel222', 'Schwefel223', 'Schwefel225',
           'Schwefel226', 'Sphere', 'Step', 'Step2', 'Step3', 'StepInt', 'Stochastic', 'StrechedVSineWave',
           'StyblinskiTang', 'SumSquares', 'Trid', 'Trigonometric1', 'Trigonometric2', 'TypeI', 'TypeII', 'Vincent',
           'WWavy', 'Weierstrass', 'Whitley', 'XinSheYang1', 'XinSheYang2', 'XinSheYang3', 'XinSheYang4', 'YaoLiu09',
           'Zakharov', 'ZeroSum']


# %% BASIC FUNCTION CLASS
class OptimalBasic:
    """
    This is the basic class for a generic optimisation problem.
    """

    def __init__(self, variable_num=2):
        """
        Initialise a problem object using only the dimensionality of its domain.

        :param int variable_num: optional.
            Number of dimensions or variables for the problem domain. The default values is 2 (this is the common option
            for plotting purposes).
        """
        self.variable_num = variable_num
        self.max_search_range = np.array([0] * self.variable_num)
        self.min_search_range = np.array([0] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}
        self.plot_object = None
        self.func_name = ''
        self.save_dir = '{0}/function_plots/'.format(os.path.dirname(os.path.abspath(__file__)))

        self.__offset_domain = 0.0
        self.__scale_domain = 1.0
        self.__scale_function = 1.0
        self.__offset_function = 0.0
        self.__noise_type = 'uniform'
        self.__noise_level = 0.0

    def get_features(self, fmt='string', wrd='1', fts=None):
        """
        Return the categorical features of the current function.

        :param str fmt: optional
            Format to deliver the features. Possible options are 'latex' and 'string'. If none of these options are
            chosen, this method returns the equivalent decimal value of the binary sequence corresponding to the
            features, e.g., 010 -> 2. The default is 'string'.
        :param str wrd: optional
            Specification to represent the features. Possible values are 'Yes' (for 'Yes' or 'No'), '1' (for '1' or
            '0'), and 'X' (for 'X' or ' '). If none of these options are chosen, features are represented as binary
            integers, i.e., 1 or 0. The default is '1'.
        :param list fts: optional
            Features to be read. The available features are: 'Continuous', 'Differentiable', 'Separable', 'Scalable',
            'Unimodal', 'Convex' The default is ['Differentiable', 'Separable', 'Unimodal'].

        :return: str or int
        """
        # Default features to deliver
        if fts is None:
            fts = ['Differentiable', 'Separable', 'Unimodal']

        def translate_conditional(value):
            if wrd == "Yes":
                words = ["Yes", "No"]
            elif wrd == "1":
                words = ["1", "0"]
            elif wrd == "X":
                words = ["X", " "]
            else:
                words = [1, 0]
            return words[0] if value else words[1]

        # Get the list of features as strings
        features = [translate_conditional(self.features[key]) for key in fts]

        # Return the list according to the format specified
        if fmt == 'latex':
            return " & ".join(features)
        elif fmt == 'string':
            return "".join(features)
        else:
            return sum(features)

    def set_offset_domain(self, value=None):
        """
        Add an offset value for the problem domain, i.e., f(x + offset).

        :param float value:
            The value to add to the variable before evaluate the function. It could be a float or numpy.array. The
            default is None.
        """
        if value:
            self.__offset_domain = value

    def set_offset_function(self, value=None):
        """
        Add an offset value for the problem function, i.e., f(x) + offset

        :param float value:
            The value to add to the function after evaluate it. The default is None.
        """
        if value:
            self.__offset_function = value

    def set_scale_domain(self, value=None):
        """
        Add a scale value for the problem domain, i.e., f(scale * x)

        :param float value:
            The value to add to the variable before evaluate the function. It could be a float or numpy.array. The
            default is None.
        """
        if value:
            self.__scale_domain = value

    def set_scale_function(self, value=None):
        """
        Add a scale value for the problem function, i.e., scale * f(x)

        :param float value:
            The value to add to the function after evaluate it. The default is None.
        """
        if value:
            self.__scale_function = value

    def set_noise_type(self, noise_distribution=None):
        """
        Specify the noise distribution to add, i.e., f(x) + noise

        :param str noise_distribution:
            Noise distribution. It can be 'gaussian' or 'uniform'. The default is None.
        """
        if noise_distribution:
            self.__noise_type = noise_distribution

    def set_noise_level(self, value=None):
        """
        Specify the noise level, i.e., f(x) + value * noise

        :param float value:
            Noise level. The default is None.
        """
        if value:
            self.__noise_level = value

    def get_global_optimum_solution(self):
        """
        Return the theoretical global optimum value.

        **Note:** Not all the functions have recognised theoretical optima.

        :return: float
        """
        return self.global_optimum_solution

    def get_optimal_solution(self):
        """
        Return the theoretical solution.

        **Note:** Not all the functions have recognised theoretical optima.

        :return: numpy.array
        """
        return self.optimal_solution

    def get_search_range(self):
        """
        Return the problem domain defined by the lower and upper boundaries, both are 1-by-variable_num arrays.

        :returns: numpy.array, numpy.array
        """
        return self.min_search_range, self.max_search_range

    def set_search_range(self, min_search_range, max_search_range):
        """
        Define the problem domain defined by the lower and upper boundaries. They could be 1-by-variable_num arrays or
        floats.

        :param min_search_range:
            Lower boundary of the problem domain. It can be a numpy.array or a float.
        :param max_search_range:
            Upper boundary of the problem domain. It can be a numpy.array or a float.

        :return: None.
        """
        if isinstance(min_search_range, (float, int)) and isinstance(max_search_range, (float, int)):
            self.min_search_range = np.array([min_search_range] * self.variable_num)
            self.max_search_range = np.array([max_search_range] * self.variable_num)
        else:
            if (len(min_search_range) == self.variable_num) and (len(max_search_range) == self.variable_num):
                self.min_search_range = min_search_range
                self.max_search_range = max_search_range
            else:
                print('Invalid range!')

    def get_func_val(self, variables, *args):
        """
        Evaluate the problem function without considering additions like noise, offset, etc.

        :param numpy.array variables:
            The position where the problem function is going to be evaluated.

        :param args:
            Additional arguments that some problem functions could consider.

        :return: float
        """
        return -1

    def get_function_value(self, variables, *args):
        """
        Evaluate the problem function considering additions like noise, offset, etc. This method calls ``get_func_val``.

        :param numpy.array variables:
            The position where the problem function is going to be evaluated.

        :param args:
            Additional arguments that some problem functions could consider.

        :return: float
        """
        # Apply modifications to the position
        variables = self.__scale_domain * variables + self.__offset_domain

        # Check which kind of noise to use
        if self.__noise_type in ['gaussian', 'normal', 'gauss']:
            noise_value = np.random.randn()
        else:
            noise_value = np.random.rand()

        # Call ``get_func_val``with the modificaitons
        return self.__scale_function * self.get_func_val(variables, *args) + \
               (self.__noise_level * noise_value) + self.__offset_function

    def plot(self, samples=55, resolution=100):
        """
        Plot the current problem in 2D.

        :param int samples: Optional.
            Number of samples per dimension. The default is 55.

        :param int resolution: Optional.
            Resolution in dpi according to matplotlib.pyplot.figure(). The default is 100.

        :return: matplotlib.pyplot
        """
        # Generate the samples for each dimension.
        x = np.linspace(self.min_search_range[0], self.max_search_range[0], samples)
        y = np.linspace(self.min_search_range[1], self.max_search_range[1], samples)

        # Create the grid matrices
        matrix_x, matrix_y = np.meshgrid(x, y)

        # Evaluate each node of the grid into the problem function
        matrix_z = []
        for xy_list in zip(matrix_x, matrix_y):
            z = []
            for xy_input in zip(xy_list[0], xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[2:self.variable_num]))
                z.append(self.get_function_value(np.array(tmp)))
            matrix_z.append(z)
        matrix_z = np.array(matrix_z)

        # Initialise the figure
        fig = plt.figure(figsize=[4, 3], dpi=resolution, facecolor='w')
        ls = LightSource(azdeg=90, altdeg=45)
        rgb = ls.shade(matrix_z, plt.cm.jet)

        # Plot data
        ax = fig.gca(projection='3d', proj_type='ortho')
        ax.plot_surface(matrix_x, matrix_y, matrix_z, rstride=1, cstride=1, linewidth=0.5,
                        antialiased=False, facecolors=rgb)  #

        # Adjust the labels
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x, y)$')
        ax.set_zlabel('$f(x, y)$')
        ax.set_title(self.func_name)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Set last adjustments
        self.plot_object = plt.gcf()
        plt.grid(linewidth=1.0)
        plt.gcf().subplots_adjust(left=0.05, right=0.85)
        plt.show()

        # Return the object for further modifications or for saving
        return self.plot_object

    # TODO: Improve function to generate better images
    def save_fig(self, samples=100, resolution=333, ext='png'):
        """
        Save the 2D representation of the problem function. There is no requirement to plot it before.

        :param int samples: Optional.
            Number of samples per dimension. The default is 100.
        :param int resolution: Optional.
            Resolution in dpi according to matplotlib.pyplot.figure(). The default is 333.
        :param str ext: Optional.
            Extension of the image file. The default is 'png'

        :return: None.
        """
        # Verify if the path exists
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        # Verify if the figure was previously plotted. If not, then do it
        if self.plot_object is None:
            self.plot(samples, resolution)

        # Save it
        plt.tight_layout()
        self.plot_object.savefig(self.save_dir + self.func_name + '.' + ext)
        plt.show()


# %% SPECIFIC PROBLEM FUNCTIONS
# 1 - Class Ackley 1 function
class Ackley1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([35.] * self.variable_num)
        self.min_search_range = np.array([-35.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Ackley 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, a=20., b=0.2, c=2. * np.pi):
        return a + np.e - (a * np.exp(-b * np.sqrt(
            np.sum(np.square(variables)) / self.variable_num)) +
                           np.exp(np.sum(np.cos(c * variables)) / self.variable_num))


# 4 - Class Ackley 4 function
class Ackley4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([35] * self.variable_num)
        self.min_search_range = np.array([-35] * self.variable_num)
        self.optimal_solution = np.array([-1.479252, -0.739807])
        self.global_optimum_solution = -3.917275
        self.func_name = 'Ackley 4'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.exp(-0.2) * np.sum(np.sqrt(
            np.square(variables[:-1]) + np.square(variables[1:]))) + \
               3. * np.sum(np.cos(2. * variables[:-1]) +
                           np.sin(2. * variables[:-1]))


# 6 - Class Alpine 1 function
class Alpine1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Alpine 1'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.abs(variables * np.sin(variables) + 0.1 * variables))


# 7 - Class Alpine 2 function
class Alpine2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([7.917] * self.variable_num)
        self.global_optimum_solution = 2.808 ** self.variable_num
        self.func_name = 'Alpine 2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.prod(np.sqrt(variables) * np.sin(variables))


# 25 - Class Brown function
class Brown(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([4.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Brown'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        xi = np.square(variables[:-1])
        xi1 = np.square(variables[1:])
        return np.sum(np.power(xi, xi1 + 1.) + np.power(xi1, xi + 1.))


# Class Brent function
class Brent(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([0.] * self.variable_num)
        self.min_search_range = np.array([-20.] * self.variable_num)
        self.optimal_solution = np.array([10.] * self.variable_num)
        self.global_optimum_solution = np.exp(-self.variable_num * 100.)
        self.func_name = 'Brent'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(variables + 10.)) + np.exp(
            -np.sum(np.square(variables)))


# 34 - Class Chung Reynolds function [Al-Roomi2015]
class ChungReynolds(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Chung Reynolds'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.square(np.sum(np.square(variables)))


# 40 - Class Csendes function
class Csendes(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([2.] * self.variable_num)
        self.min_search_range = np.array([-2.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Csendes'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.power(variables, 6.) * (2. + np.sin(1 / variables))) \
            if np.prod(variables) != 0 else 0.


# 38 - Class Cosine Mixture function
class CosineMixture(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.1 * self.variable_num
        self.func_name = 'Cosine Mixture'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0.1 * np.sum(np.cos(5. * np.pi * variables)) - np.sum(np.square(variables))


# 43 - Class Deb 1 function
class Deb1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([-0.1] * self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Deb 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return -np.sum(np.power(np.sin(5. * np.pi * variables), 6.)) / self.variable_num


# Class Deb 2 function [https://al-roomi.org/benchmarks/unconstrained/n-dimensions/232-deb-s-function-no-02]
class Deb2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([np.power(1. / 10. + 0.05, 4. / 3.)] * self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Deb 2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.power(np.sin(5. * np.pi * (
                np.power(variables, 3. / 4.) - 0.05)), 6.)) / (-self.variable_num)


# 48 - Class Dixon & Price function
class DixonPrice(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.power(2., np.power(
            2., - np.arange(self.variable_num)) - 1.)
        self.global_optimum_solution = 0.
        self.func_name = 'Dixon-Price'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.square(variables[0] - 1.) + np.sum([
            i * np.square(2. * np.square(variables[i]) - variables[i - 1])
            for i in range(1, self.variable_num)])


# Class Drop-Wave function [http://benchmarkfcns.xyz/benchmarkfcns/dropwavefcn.html]
class DropWave(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Drop Wave'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return -(1. + np.cos(12. * np.linalg.norm(variables))) / (
                0.5 * np.sum(np.square(variables)) + 2.)


# 53 - Class Egg Holder function
class EggHolder(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([512.] * self.variable_num)
        self.min_search_range = np.array([-512.] * self.variable_num)
        self.optimal_solution = np.array([512., 404.2319])
        self.global_optimum_solution = -959.6407
        self.func_name = 'Egg Holder'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        xi = variables[:-1]
        xi1 = variables[1:]
        return np.sum(-(xi1 + 47.) * np.sin(np.sqrt(
            np.abs(xi1 + xi / 2. + 47.))) - xi * np.sin(np.sqrt(
            np.abs(xi - xi1 - 47.))))


# Class Expanded Two-Peak Trap function [Qu2016]
class ExpandedTwoPeakTrap(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Expanded Two-Peak Trap'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        def get_cases(y):
            if y < 0.:
                return -160. + np.square(y)
            elif 0. <= y < 15.:
                return 160. * (y - 15.) / 15.
            elif 15. <= y < 20.:
                return 200. * (15. - y) / 5.
            else:
                return -200. + np.square(y - 20.)

        return np.sum(np.vectorize(get_cases)(variables + 20.)) + 200. * \
               self.variable_num


# Class Expanded Five-Uneven-Peak Trap function [Qu2016]
class ExpandedFiveUnevenPeakTrap(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Expanded Five-Uneven-Peak Trap'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        def get_cases(x):
            if x < 0.:
                return -200. + np.square(x)
            elif 0. <= x < 2.5:
                return -80. * (2.5 - x)
            elif 2.5 <= x < 5.:
                return -64. * (x - 2.5)
            elif 5. <= x < 7.5:
                return -64. * (7.5 - x)
            elif 7.5 <= x < 12.5:
                return -28. * (x - 7.5)
            elif 12.5 <= x < 17.5:
                return -28. * (17.5 - x)
            elif 17.5 <= x < 22.5:
                return -32. * (x - 17.5)
            elif 22.5 <= x < 27.5:
                return -32. * (27.5 - x)
            elif 27.5 <= x < 30.:
                return -80. * (x - 27.5)
            else:
                return -200. + np.square(x - 30.)

        return np.sum(np.vectorize(get_cases)(variables)) + 200. * self.variable_num


# Class Expanded Equal Minima function [Qu2016]
class ExpandedEqualMinima(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Expanded Equal Minima'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        def get_cases(y):
            if 0. <= y < 1.:
                return -np.power(np.sin(5. * np.pi * y), 6.)
            else:
                return np.square(y)

        return np.sum(np.vectorize(get_cases)(variables + 0.1)) + self.variable_num


# Class Expanded Decreasing Minima function [Qu2016]
class ExpandedDecreasingMinima(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Expanded Decreasing Minima'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        def get_cases(y):
            if 0. <= y < 1.:
                return -np.exp(-2. * np.log(2.) * np.square((y - 0.1) / 0.8)) * np.power(np.sin(
                    5. * np.pi * y), 6.)
            else:
                return np.square(y)

        return np.sum(np.vectorize(get_cases)(variables + 0.1)) + \
               self.variable_num


# Class Expanded Uneven Minima function [Qu2016]
class ExpandedUnevenMinima(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Expanded Uneven Minima'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        def get_cases(y):
            if 0. <= y < 1.:
                return -np.power(np.sin(5. * np.pi * (np.power(y, 3. / 4.) - 0.05)), 6.)
            else:
                return np.square(y)

        return np.sum(np.vectorize(get_cases)(variables + 0.079699392688696)) - \
               self.variable_num


# Class Vincent function [http://infinity77.net/global_optimization/test_functions_nd_V.html#go_benchmark
# .VenterSobiezcczanskiSobieski]
class Vincent(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([0.25] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Vincent'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return -np.sum(np.sin(10. * np.log10(variables)))


# Class Modified Vincent function [Qu2016]
class ModifiedVincent(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Modified Vincent'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        def get_cases(y):
            if y < 0.25:
                return np.square(0.25 - y) - np.sin(10. * np.log(2.5))
            elif 0.25 <= y <= 10.:
                return np.sin(10. * np.log(y))
            else:
                return np.square(y - 10.) - np.sin(10. * np.log(10))

        return np.sum(np.vectorize(get_cases)(variables + 4.1112) + 1.0) / \
               self.variable_num


# 54 - Class Exponential function
class Exponential(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 1.
        self.func_name = 'Exponential'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return -np.exp(-0.5 * np.sum(np.square(variables)))


# Class Giunta function
class Giunta(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([0.4673200277395354] * self.variable_num)
        self.global_optimum_solution = 0.06447042053690566
        self.func_name = 'Giunta'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0.6 + np.sum(np.square(np.sin(1. - (16. / 15.) * variables)) - (1. / 50.)
                            * np.sin(4. - (64. / 15.) * variables)
                            - np.sin(1. - (16. / 15.) * variables))


# 59 - Class Griewank function
class Griewank(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Griewank'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(variables)) / 4000. - np.prod(np.cos(
            variables / np.sqrt(np.arange(self.variable_num) + 1.))) + 1.


# Class Happy Cat function [http://benchmarkfcns.xyz/benchmarkfcns/happycatfcn.html]
class HappyCat(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([-1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Happy Cat'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, alpha=1. / 8., *args):
        x1 = np.sum(variables)
        x2 = np.sum(np.square(variables))
        return np.power(np.square(x2 - self.variable_num), alpha) + (0.5 * x2 + x1) / (
            self.variable_num) + 0.5


# 74 - Class Mishra 1 function
class Mishra1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 2.
        self.func_name = 'Mishra 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        g_funct = self.variable_num - np.sum(variables[:-1])
        return np.power(1. + g_funct, g_funct)


# 75 - Class Mishra 2 function
class Mishra2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 2.
        self.func_name = 'Mishra 2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        g_funct = self.variable_num - 0.5 * np.sum(
            variables[:-1] + variables[1:])
        return np.power(1 + g_funct, g_funct)


# 80 - Class Mishra 7 function
class Mishra7(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([np.power(np.math.factorial(
            self.variable_num), 1. / self.variable_num)] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Mishra 7'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.square(np.prod(variables) - np.math.factorial(
            self.variable_num))


# 84 - Class Mishra 11 function (Arithmetic Mean - Geometric Mean Equality)
class Mishra11(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([np.nan] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Mishra 11'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.square(np.sum(np.abs(variables)) / self.variable_num -
                         np.power(np.prod(np.abs(variables)),
                                  1. / self.variable_num))


# Class Needle-Eye function [Al-Roomi2015]
class NeedleEye(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 1.
        self.func_name = 'Needle-Eye'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, eye=0.0001, *args):
        x = np.abs(variables)
        t = np.heaviside(x - eye, 1.)
        return 1. if np.all(x < eye) else np.sum((100. + x) * t)


# 87 - Class Pathological function
class Pathological(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Pathological'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        xi = variables[:-1]
        xj = variables[1:]
        return np.sum(0.5 + np.square(np.sin(np.sqrt(
            100. * np.square(xi) + np.square(xj))) - 0.5) / (1. + 0.001 * np.power(xi - xj, 4.)))


# 89 - Class Pinter function
class Pinter(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Pinter'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        component_a = variables[:-2] * np.sin(variables[1:-1]) + np.sin(variables[2:])
        component_b = np.square(variables[:-2]) - 2. * variables[1:-1] + (
                3. * variables[2:] - np.cos(variables[1:-1]) + 1.)
        i = np.arange(self.variable_num) + 1.
        return np.sum(i * np.square(variables)) + np.sum(
            20. * i[1:-1] * np.square(np.sin(component_a))) + np.sum(
            i[1:-1] * np.log10(1. + i[1:-1] * np.square(component_b)))


# Class Periodic function
class Periodic(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.9
        self.func_name = 'Periodic'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 1. + np.sum(np.square(np.sin(variables))) - 0.1 * np.exp(
            -np.sum(np.square(variables)))


# 93 - Class Powell Sum function
class PowellSum(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Powel Sum'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.power(np.abs(variables), np.arange(self.variable_num) + 2.))


# 98 - Class Qing function
class Qing(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.] * self.variable_num)
        self.min_search_range = np.array([-500.] * self.variable_num)
        self.optimal_solution = np.array(np.sqrt(np.arange(
            self.variable_num) + 1.))
        self.global_optimum_solution = 0.
        self.func_name = 'Qing'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(np.square(variables) - (np.arange(self.variable_num) + 1.)))


# 100 - Class Quartic function
class Quartic(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.28] * self.variable_num)
        self.min_search_range = np.array([-1.28] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Quartic'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.power(variables, 4.) * (np.arange(self.variable_num) + 1.))


# 101 - Class Quintic function
class Quintic(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([-1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Quintic'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.abs(np.power(variables, 5.) - 3. * np.power(variables, 4.)
                             + 4. * np.power(variables, 3.)
                             + 2. * np.power(variables, 2.) - 10. * variables - 4.))


# 102 - Class Rana function
class Rana(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.000001] * self.variable_num)
        self.min_search_range = np.array([-500.000001] * self.variable_num)
        self.optimal_solution = np.array([-500.] * self.variable_num)
        self.global_optimum_solution = -511.70430 * self.variable_num + 511.68714
        self.func_name = 'Rana'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        t1 = np.sqrt(np.abs(variables[1:] + variables[:-1] + 1.))
        t2 = np.sqrt(np.abs(variables[1:] - variables[:-1] + 1.))
        return np.sum((variables[1:] + 1.) * np.cos(t1) * np.sin(t1)
                      + variables[:-1] * np.cos(t1) * np.sin(t2))


# Class Rastrigin function
class Rastrigin(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Rastrigin'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 10. * self.variable_num + np.sum(
            np.square(variables) - 10. * np.cos(2. * np.pi * variables))


# Class Ridge function
class Ridge(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.] * self.variable_num)
        self.min_search_range = np.array([-5.] * self.variable_num)
        self.optimal_solution = np.array([self.min_search_range[0], *[0.] * (self.variable_num - 1)])
        self.global_optimum_solution = 0.
        self.func_name = 'Ridge'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, d=2., alpha=0.1):
        return variables[0] + d * np.power(np.sum(np.square(variables)), alpha)


# 105 - Class Rosenbrock function
class Rosenbrock(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([30.] * self.variable_num)
        self.min_search_range = np.array([-30.] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Rosenbrock'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(100. * np.square(variables[1:] - np.square(
            variables[:-1])) + np.square(variables[:-1] - 1))


# 110 - Class Salomon function
class Salomon(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Salomon'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 1. - np.cos(2. * np.pi * np.sqrt(np.sum(np.square(variables)))) + \
               0.1 * np.sqrt(np.sum(np.square(variables)))


# 111 - Class Sargan function
class Sargan(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Sargan'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return self.variable_num * np.sum(np.square(variables) + 0.4 * np.sum(
            np.multiply(1. - np.identity(self.variable_num), np.outer(
                variables, variables)), 0))


# 117 - Class Schumer-Steiglitz function [Al-Roomi2015]
class SchumerSteiglitz(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schumer Steiglitz'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.power(variables, 4.))


# 118 - Class Schwefel function
class Schwefel(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, alpha=0.5):
        return np.power(np.sum(np.square(variables)), alpha)


# 119 - Class Schwefel 1.2 function
class Schwefel12(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel 1.2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(np.sum(np.tril(np.outer(np.ones(self.variable_num),
                                                        variables)), 1)))


# 120 - Class Schwefel 2.04 function
class Schwefel204(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel 2.04'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(variables - 1.) + np.square(variables[0] - np.square(variables)))


# 122 - Class Schwefel 2.20 function
class Schwefel220(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel 2.20'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.abs(variables))


# 123 - Class Schwefel 2.21 function
class Schwefel221(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel 2.21'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.max(np.abs(variables))


# 124 - Class Schwefel 2.22 function
class Schwefel222(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel 2.22'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.abs(variables)) + np.prod(np.abs(variables))


# 125 - Class Schwefel 2.23 function
class Schwefel223(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel 2.23'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.power(variables, 10.))


# 127 - Class Schwefel 2.25 function
class Schwefel225(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schwefel 2.25'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(variables[1:] - 1.) + np.square(
            variables[0] - np.square(variables[1:])))


# 128 - Class Schwefel 2.26 function [http://benchmarkfcns.xyz/benchmarkfcns/schwefelfcn.html]
class Schwefel226(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.] * self.variable_num)
        self.min_search_range = np.array([-500.] * self.variable_num)
        self.optimal_solution = np.array([np.square(np.pi * 1.5)] * self.variable_num)
        self.global_optimum_solution = -418.983
        self.func_name = 'Schwefel 2.26'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(variables * np.sin(np.abs(variables))) / (
            -self.variable_num)


# 133 - Class Schubert function
class Schubert(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([np.nan] * self.variable_num)
        self.global_optimum_solution = -186.7309
        self.func_name = 'Schubert'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        j = np.arange(1, 6)
        return np.prod(np.sum(np.cos(np.outer(variables, j + 1) + np.outer(
            np.ones(self.variable_num), j)), 1))


# 134 - Class Schubert 3 function
class Schubert3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([np.nan] * self.variable_num)
        self.global_optimum_solution = -29.6733337
        self.func_name = 'Schubert 3'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        j = np.arange(1, 6)
        return np.sum(np.sum(np.outer(np.ones(self.variable_num), j) *
                             np.sin(np.outer(variables, j + 1) + np.outer(
                                 np.ones(self.variable_num), j)), 1))


# 135 - Class Schubert 4 function
class Schubert4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([np.nan] * self.variable_num)
        self.global_optimum_solution = -25.740858
        self.func_name = 'Schubert 4'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        j = np.arange(1, 6)
        return np.sum(np.sum(np.outer(np.ones(self.variable_num), j) *
                             np.cos(np.outer(variables, j + 1) + np.outer(
                                 np.ones(self.variable_num), j)), 1))


# 136 - Class Schaffer N6 function
class SchafferN6(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schaffer N6'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0.5 * (self.variable_num - 1.) + np.sum((np.square(
            np.sin(np.sqrt(np.square(variables[:-1]) + np.square(
                variables[1:])))) - 0.5) / np.square(1. + 0.001 * (
                variables[:-1] + variables[1:])))


# 136* - Class Schaffer N1 function [http://benchmarkfcns.xyz/benchmarkfcns/schaffern1fcn.html]
class SchafferN1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schaffer N1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0.5 * (self.variable_num - 1.) + np.sum((np.square(
            np.sin(np.square(np.square(variables[:-1]) + np.square(
                variables[1:])))) - 0.5) / np.square(1. + 0.001 * (
                variables[:-1] + variables[1:])))


# 136* - Class Schaffer N2 function [http://benchmarkfcns.xyz/benchmarkfcns/schaffern2fcn.html]
class SchafferN2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schaffer N2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0.5 * (self.variable_num - 1.) + np.sum((np.square(np.sin(
            np.square(variables[:-1]) + np.square(variables[1:]))) - 0.5) / np.square(
            1. + 0.001 * (variables[:-1] + variables[1:])))


# 136* - Class Schaffer N3 function [http://benchmarkfcns.xyz/benchmarkfcns/schaffern3fcn.html]
class SchafferN3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schaffer N3'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        x2 = np.square(variables[:-1])
        y2 = np.square(variables[1:])
        return 0.5 * (self.variable_num - 1.) + np.sum((np.square(np.sin(np.cos(
            np.abs(x2 + y2)))) - 0.5) / np.square(1. + 0.001 * (x2 + y2)))


# 136* - Class Schaffer N4 function [http://benchmarkfcns.xyz/benchmarkfcns/schaffern4fcn.html]
class SchafferN4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Schaffer N4'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        x2 = np.square(variables[:-1])
        y2 = np.square(variables[1:])
        return 0.5 * (self.variable_num - 1.) + np.sum((np.square(np.cos(np.sin(
            np.abs(x2 + y2)))) - 0.5) / np.square(1. + 0.001 * (x2 + y2)))


# 137 - Class Sphere function
class Sphere(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Sphere'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(variables))


# 138 - Class Step function
class Step(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Step'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.floor(np.abs(variables)))


# 139 - Class Step 2 function
class Step2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.5] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Step 2'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(np.floor(variables + 0.5)))


# 140 - Class Step 3 function
class Step3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Step 3'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.floor(np.square(variables)))


# 141 - Class Step Int function
class StepInt(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([-5.12] * self.variable_num)
        self.global_optimum_solution = 25. - 6. * self.variable_num
        self.func_name = 'Step Int'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.floor(variables)) + 25


# 142 - Class Streched V Sine Wave function
class StrechedVSineWave(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Streched V Sine Wave'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        f10 = np.power(np.square(variables[:-1]) + np.square(variables[1:]), 0.10)
        f25 = np.power(np.square(variables[:-1]) + np.square(variables[1:]), 0.25)
        return np.sum(f25 * (np.square(np.sin(f10)) + 0.1))


# 143 - Class Sum Squares function
class SumSquares(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Sum Squares'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.arange(1, self.variable_num + 1) * np.square(variables))


# 150 - Class Trid function [http://www.sfu.ca/~ssurjano/trid.html]
class Trid(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.square(self.variable_num)] * self.variable_num)
        self.min_search_range = np.array([-np.square(self.variable_num)] * self.variable_num)
        self.optimal_solution = np.array(np.arange(1., self.variable_num + 1.) * (
                self.variable_num + 1. - np.arange(1., self.variable_num + 1.)))
        self.global_optimum_solution = -self.variable_num * (
                self.variable_num + 4.) * (self.variable_num - 1.) / 6.
        self.func_name = 'Trid'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(variables - 1)) - np.sum(variables[1:] * variables[:-1])


# 153 - Class Trigonometric 1 function
class Trigonometric1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Trigonometric 1'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        x = np.outer(variables, np.ones(self.variable_num))
        i = np.outer(np.arange(1, self.variable_num + 1), np.ones(self.variable_num))
        return np.sum(np.square(self.variable_num - np.sum(np.cos(x) + i * (
                1 - np.cos(x.T) - np.sin(x.T)), 0)))


# 154 - Class Trigonometric 2 function
class Trigonometric2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.] * self.variable_num)
        self.min_search_range = np.array([-500.] * self.variable_num)
        self.optimal_solution = np.array([0.9] * self.variable_num)
        self.global_optimum_solution = 1.
        self.func_name = 'Trigonometric 2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 1 + np.sum(8. * np.square(np.sin(7. * np.square(
            variables - 0.9))) + 6. * np.square(np.sin(14. * np.square(
            variables - 0.9))) + np.square(variables - 0.9))


# 165 - Class W-Wavy function
class WWavy(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi] * self.variable_num)
        self.min_search_range = np.array([-np.pi] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'W-Wavy'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, k=10., *args):
        return 1. - np.sum(np.cos(k * variables) * np.exp(-0.5 * np.square(variables))) / \
               self.variable_num


# 166 - Class Weierstrass function
class Weierstrass(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([0.5] * self.variable_num)
        self.min_search_range = np.array([-0.5] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Weierstrass'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, kmax=20, a=0.5, b=3., *args):
        x = np.outer(variables + 0.5, np.ones(kmax + 1))
        k = np.outer(np.ones(self.variable_num), np.arange(kmax + 1))
        return np.sum(np.sum(np.power(a, k) * np.cos(2. * np.pi * np.power(b, k) * x), 1)
                      - self.variable_num * np.sum(np.power(a, k) * np.cos(np.pi
                                                                           * np.power(b, k)), 1))


# 167 - Class Whitley function
class Whitley(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.24] * self.variable_num)
        self.min_search_range = np.array([-10.24] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Whitley'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        x = np.outer(variables, np.ones(self.variable_num))
        matrix_x = 100. * np.square(np.square(x) - x.T) + np.square(1. - x.T)
        return np.sum(np.square(matrix_x) / 4000. - np.cos(matrix_x + 1.))


# 169 - Class Xin-She Yang 1 function
class XinSheYang1(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.] * self.variable_num)
        self.min_search_range = np.array([-5.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Xin-She Yang 1'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.random.rand(self.variable_num) * np.power(np.abs(
            variables), np.arange(1, self.variable_num + 1)))


# 170 - Class Xin-She Yang 2 function
class XinSheYang2(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([2. * np.pi] * self.variable_num)
        self.min_search_range = np.array([-2. * np.pi] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Xin-She Yang 2'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.abs(variables)) * np.exp(-np.sum(np.sin(np.square(variables))))


# 171 - Class Xin-She Yang 3 function
class XinSheYang3(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([20.] * self.variable_num)
        self.min_search_range = np.array([-20.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Xin-She Yang 3'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, m=5., beta=15., *args):
        return np.exp(-np.sum(np.power(variables / beta, 2. * m))) - 2. \
               * np.exp(-np.sum(np.square(variables))) \
               * np.prod(np.square(np.cos(variables)))


# 172 - Class Xin-She Yang 4 function
class XinSheYang4(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Xin-She Yang 4'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return (np.sum(np.square(np.sin(variables))) - np.exp(-np.sum(
            np.square(variables)))) * np.exp(-np.sum(np.square(
            np.sin(np.abs(variables)))))


# 173 - Class Zakharov function
class Zakharov(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-5.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Zakharov'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        ixi = np.arange(1, self.variable_num + 1) * variables
        return np.sum(np.square(variables)) + np.square(0.5 * np.sum(ixi)) + np.power(0.5 * np.sum(ixi), 4.)


# Class Styblinski-Tang function [http://benchmarkfcns.xyz/benchmarkfcns/styblinskitankfcn.html]
class StyblinskiTang(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.] * self.variable_num)
        self.min_search_range = np.array([-5.] * self.variable_num)
        self.optimal_solution = np.array([-2.903534] * self.variable_num)
        self.global_optimum_solution = -39.16599 * self.variable_num
        self.func_name = 'Styblinski-Tang'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0.5 * np.sum(np.power(variables, 4) - 16. * np.square(variables) + 5. * variables)


# Class Stochastic function [https://al-roomi.org/benchmarks/unconstrained/n-dimensions/267-xin-she-yang-s-function
# -no-07]
class Stochastic(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.] * self.variable_num)
        self.min_search_range = np.array([-5.] * self.variable_num)
        self.optimal_solution = 1. / (np.arange(self.variable_num) + 1.)
        self.global_optimum_solution = 0.
        self.func_name = 'Stochastic'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.random.rand(self.variable_num) * abs(variables - 1. / (
                np.arange(self.variable_num) + 1.)))


# Class Ellipsoid function \cite{Finck2009}
class Ellipsoid(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Ellipsoid'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(np.power(10., np.arange(self.variable_num) / (
                self.variable_num - 1.)) * variables))


# Class Hyper-Ellipsoid function http://www.geatbx.com/docu/fcnindex-01.html#P109_4163
class HyperEllipsoid(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Hyper-Ellipsoid'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.arange(1, self.variable_num + 1) * np.square(variables))


# Class Rotated-Hyper-Ellipsoid function http://www.sfu.ca/~ssurjano/rothyp.html
class RotatedHyperEllipsoid(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([65.536] * self.variable_num)
        self.min_search_range = np.array([-65.536] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Rotated-Hyper-Ellipsoid'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.sum(np.tril(np.outer(np.ones(self.variable_num), np.square(variables))))


# Class Michalewicz function \cite{Molga2005}
class Michalewicz(OptimalBasic):
    # Optimal solution for the first 100 dimensions (approximated), with m = 10
    approximated_optima = [2.2029, 1.5708, 1.2850, 1.9231, 0.9967, 2.3988, 1.8772, 0.7885,
                           1.9590, 1.2170, 1.1604, 2.1268, 0.6188, 1.5708, 1.9023, 1.2419,
                           1.9426, 1.1709, 2.2214, 1.9238, 0.4870, 1.9527, 1.2256, 1.1998,
                           1.9366, 0.7549, 1.9591, 2.7528, 0.7148, 1.9451, 1.1970, 1.0390,
                           1.9335, 0.3828, 2.6815, 1.5265, 1.2113, 1.9406, 1.1798, 2.3032,
                           1.8027, 0.7666, 2.1156, 0.7490, 0.7406, 1.9377, 0.7247, 2.1026,
                           2.3103, 1.0420, 1.9426, 1.4117, 0.9155, 2.3994, 0.3010, 1.9466,
                           1.2826, 1.2027, 1.9401, 0.7588, 2.6833, 1.5708, 0.7405, 1.9438,
                           1.2010, 1.1919, 1.9381, 0.4668, 1.9469, 2.0046, 1.0211, 1.9416,
                           1.1333, 2.0497, 1.7956, 0.4416, 2.0094, 1.2063, 1.1986, 1.9398,
                           0.7405, 2.1526, 2.4015, 0.6413, 1.9426, 1.0977, 1.1422, 1.9092,
                           0.2366, 2.3994, 1.0672, 1.2034, 1.9410, 1.1906, 2.3131, 1.7996,
                           0.7481, 2.1170, 1.2431, 1.1963]

    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array(self.approximated_optima[:self.variable_num])
        self.global_optimum_solution = self.get_function_value(self.optimal_solution)
        self.func_name = 'Michalewicz'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, m=10.):
        return -np.sum(np.sin(variables) * np.power(np.sin(np.arange(
            1, self.variable_num + 1) * np.square(variables) / np.pi), 2. * m))


# Class K-Tablet function \cite{Sakuma2004}
class KTablet(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'K-Tablet'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        k = int(self.variable_num // 4)
        return np.sum(variables[:k]) + np.sum(np.square(100. * variables[k:]))


# Class Perm 01 function [http://infinity77.net/global_optimization/test_functions_nd_P.html#go_benchmark.PenHolder]
class Perm01(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([self.variable_num + 1] * self.variable_num)
        self.min_search_range = np.array([-self.variable_num] * self.variable_num)
        self.optimal_solution = np.arange(1, self.variable_num + 1)
        self.global_optimum_solution = 0.
        self.func_name = 'Perm 01'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, beta=1.):
        x = np.outer(variables, np.ones(self.variable_num))
        inds = np.outer(np.ones(self.variable_num), np.arange(self.variable_num) + 1.)
        return np.sum(np.square(np.sum((np.power(inds.T, inds) + beta) * (
                np.power(x / inds.T, inds) - 1.), 0)))


# Class Perm 02 function [http://www.sfu.ca/~ssurjano/perm0db.html]
class Perm02(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([self.variable_num + 1] * self.variable_num)
        self.min_search_range = np.array([-self.variable_num] * self.variable_num)
        self.optimal_solution = 1. / np.arange(1, self.variable_num + 1)
        self.global_optimum_solution = 0.
        self.func_name = 'Perm 02'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, beta=1.):
        x = np.outer(variables, np.ones(self.variable_num))
        inds = np.outer(np.ones(self.variable_num), np.arange(self.variable_num) + 1.)
        return np.sum(np.square(np.sum((inds.T + beta) * (
                np.power(x, inds) - np.power(1. / inds.T, inds)), 0)))


# Class Yao Liu 09 function [http://infinity77.net/global_optimization/test_functions_nd_Y.html#go_benchmark.YaoLiu04]
class YaoLiu09(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Yao-Liu 09'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(variables) - 10. * np.cos(2. * np.pi * variables) + 10.)


# Class Zero Sum function [http://infinity77.net/global_optimization/test_functions_nd_Z.html#go_benchmark.Zirilli]
class ZeroSum(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Zero Sum'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0. if (np.sum(variables) == 0.) else 1. + np.power(
            1.0e4 * np.abs(np.sum(variables)), 0.5)


# Class Levy function [http://www.sfu.ca/~ssurjano/levy.html]
class Levy(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([1.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Levy'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        w = 1. + (variables - 1.) / 4.
        f0 = np.square(np.sin(np.pi * w[0])) + (self.variable_num - 1.) * np.square(
            w[-1] - 1.) * (1. + np.square(np.sin(2. * np.pi * w[-1])))
        return f0 + np.sum(np.square(w[:-1] - 1.) * (1. + 10. * np.square(np.sin(
            np.pi * w[:-1] + 1.))))


# Class Price 01 function
class Price01(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.] * self.variable_num)
        self.min_search_range = np.array([-500.] * self.variable_num)
        self.optimal_solution = np.array([5.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Price 01'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return np.sum(np.square(np.abs(variables) - 5.))


# Class Bohachevsky function [http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark
# .Bohachevsky]
class Bohachevsky(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([15.] * self.variable_num)
        self.min_search_range = np.array([-15.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Bohachevsky'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        x = variables[:-1]
        y = variables[1:]
        return np.sum(np.square(x) + 2. * np.square(y) - 0.3 * np.cos(3. * np.pi * x)
                      - 0.4 * np.cos(4. * np.pi * y) + 0.7)


# Class Bohachevsky function [http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark
# .CarromTable]
class CarromTable(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([9.646157266348881] * self.variable_num)
        self.global_optimum_solution = -24.15681551650653
        self.func_name = 'Carrom Table'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return - (1. / 30.) * np.exp(2. * np.abs(1. - np.linalg.norm(variables) / np.pi)) * \
               np.prod(np.square(np.cos(variables)))


# Class CrownedCross function [http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark
# .CrownedCross]
class CrownedCross(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.0001
        self.func_name = 'Crowned Cross'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return 0.0001 * np.power(np.abs(np.exp(np.abs(100. - np.linalg.norm(
            variables) / np.pi)) * np.prod(np.sin(variables))) + 1., 0.1)


# Class CrossInTray function [http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark
# .CrownedCross]
class CrossInTray(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([15.] * self.variable_num)
        self.min_search_range = np.array([-15.] * self.variable_num)
        self.optimal_solution = np.array([1.349406608602084] * self.variable_num)
        self.global_optimum_solution = -2.062611870822739
        self.func_name = 'Cross-in-Tray'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return -0.0001 * np.power(np.abs(np.exp(np.abs(100. - np.linalg.norm(
            variables) / np.pi)) * np.prod(np.sin(variables))) + 1., 0.1)


# Class CrossLegTable function [al-roomi]
class CrossLegTable(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)
        self.min_search_range = np.array([-10.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'Cross-Leg Table'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, *args):
        return -1. / np.power(np.abs(np.exp(np.abs(100. - np.linalg.norm(
            variables) / np.pi)) * np.prod(np.sin(variables))) + 1., 0.1)


# Class Cigar function [http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CarromTable]
class Cigar(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([-100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Cigar'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def get_func_val(self, variables, *args):
        return np.square(variables[0]) + 1.e6 * np.sum(np.square(variables[1:]))


# Class Deflected Corrugated Spring function [al-roomi]
class DeflectedCorrugatedSpring(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([10.] * self.variable_num)  # 2 alpha
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([5.] * self.variable_num)  # alpha
        self.global_optimum_solution = -1.
        self.func_name = 'Deflected Corrugated Spring'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, alpha=5., k=5.):
        x = np.square(variables - alpha)
        return 0.1 * np.sum(x) - np.cos(k * np.sqrt(np.sum(x)))


# Class Katsuura function [http://infinity77.net/global_optimization/test_functions_nd_K.html#go_benchmark.Katsuura]
class Katsuura(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([100.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 1.
        self.func_name = 'Katsuura'
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, d=32):
        two_k = np.outer(np.power(2., np.arange(1, d + 1)), np.ones(self.variable_num))
        x = np.outer(np.ones(d), variables)
        return np.prod(1. + np.arange(1, self.variable_num + 1)
                       * np.sum(np.floor(two_k * x) / two_k, 0))


# Class Jennrich-Sampson function [http://infinity77.net/global_optimization/test_functions_nd_J.html#go_benchmark
# .JennrichSampson]
class JennrichSampson(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([0.257825] * self.variable_num)
        self.global_optimum_solution = 124.3621824
        self.func_name = 'Jennrich-Sampson'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, d=10):
        i = np.outer(np.arange(1, d + 1), np.ones(self.variable_num))
        x = np.outer(np.ones(d), variables)

        return np.sum(np.square(2. + 2. * np.arange(1, d + 1) - np.sum(np.exp(i * x), 1)))


# Class Lunacek's bi-Sphere function [https://al-roomi.org/benchmarks/unconstrained/n-dimensions/228-lunacek-s-bi
# -sphere-function]
class LunacekN01(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([2.5] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = "Lunacek N01"
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, mu1=2.5, d=1.):
        s = 1. - 1. / (2. * np.sqrt(self.variable_num + 20.) - 8.2)
        mu2 = -np.sqrt((np.square(mu1) - d) / s)
        return np.min([np.sum(np.square(variables - mu1)),
                       np.sum(np.square(variables - mu2)) * s + d * self.variable_num])


# Class Lunacek's bi-Rastrigin function [https://al-roomi.org/benchmarks/unconstrained/n-dimensions/229-lunacek-s-bi
# -rastrigin-function]
class LunacekN02(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([2.5] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = "Lunacek N02"
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, mu1=2.5, d=1.):
        s = 1. - 1. / (2. * np.sqrt(self.variable_num + 20.) - 8.2)
        mu2 = -np.sqrt((np.square(mu1) - d) / s)
        return np.min([np.sum(np.square(variables - mu1)),
                       np.sum(np.square(variables - mu2)) * s + d * self.variable_num]) + 10. * np.sum(
            1. - np.cos(2. * np.pi * (variables - mu1)))


# Class Type-I Simple Deceptive Problem function \cite{Suzuki2002}
class TypeI(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([0.8] * self.variable_num)  # alpha
        self.global_optimum_solution = 0.
        self.func_name = "Type-I Simple Deceptive Problem"
        self.features = {'Continuous': True,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': True,
                         'Convex': False}

    def get_func_val(self, variables, alpha=0.8, beta=1.0):
        def get_cases(y):
            if 0. <= y <= alpha:
                return alpha - y
            else:
                return (y - alpha) / (1. - alpha)

        return np.power(np.sum(np.vectorize(get_cases)(variables)) / self.variable_num, beta)


# Class Type-II Medium-Complex Deceptive Problem function \cite{Suzuki2002}
class TypeII(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([0.] * self.variable_num)
        self.optimal_solution = np.array([0.8] * self.variable_num)  # alpha and 1 - alpha
        self.global_optimum_solution = 0.
        self.func_name = "Type-II Medium-Complex Deceptive"
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': False,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, alpha=0.8, beta=1.0):
        def get_cases(y):
            if np.random.rand() <= 0.5:
                if 0. <= y <= alpha:
                    return alpha - y
                else:
                    return (y - alpha) / (1. - alpha)
            else:
                if 0. <= y <= 1. - alpha:
                    return (1. - y - alpha) / (1. - alpha)
                else:
                    return y - 1. + alpha

        return np.power(np.sum(np.vectorize(get_cases)(variables)) / self.variable_num, beta)


# Class F2 function [al-roomi]
class F2(OptimalBasic):
    l_values = [5.1, 0.5, 2.772588722239781, 0.066832364099628, 0.64]

    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.] * self.variable_num)
        self.min_search_range = np.array([-1.] * self.variable_num)
        self.optimal_solution = np.array([self.l_values[3]] * self.variable_num)
        self.global_optimum_solution = -1.
        self.func_name = 'F2'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, k=6., l_vals=None):
        if l_vals is None:
            l_vals = self.l_values
        return -np.prod(np.power(np.sin(l_vals[0] * np.pi * variables + l_vals[1]), k)
                        * np.exp(-l_vals[2] * np.square((variables - l_vals[3]) / l_vals[4])))


# Class Inverted Cosine-Wave function [al-roomi]
class InvertedCosineWave(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.] * self.variable_num)
        self.min_search_range = np.array([-5.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = -self.variable_num + 1.
        self.func_name = 'Inverted Cosine-Wave'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, k=6., *args):
        x_vals = np.square(variables[:-1]) + np.square(variables[1:]) \
                 + 0.5 * variables[1:] * variables[:-1]
        return -np.sum(np.exp(-x_vals / 8.) * np.cos(4. * np.sqrt(x_vals)))


# Class Odd Square function [al-roomi]
class OddSquare(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.] * self.variable_num)
        self.min_search_range = np.array([-5.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = -self.variable_num + 1.
        self.func_name = 'Odd Square'
        self.features = {'Continuous': False,
                         'Differentiable': False,
                         'Separable': False,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

    def get_func_val(self, variables, k=6., *args):
        x_vals = np.square(variables[:-1]) + np.square(variables[1:]) \
                 + 0.5 * variables[1:] * variables[:-1]
        return -np.sum(np.exp(-x_vals / 8.) * np.cos(4. * np.sqrt(x_vals)))


# %% TOOLS TO HANDLE THE PROBLEMS
# List all available functions
def list_functions(rnp=True, fts=None, wrd="1"):
    """

    :param rnp: return but not print if True, otherwise, print but not return
    :param fts: features to export/print. Possible options: 'Continuous', 'Differentiable','Separable', 'Scalable',
        'Unimodal', 'Convex'. Default: 'Differentiable','Separable', 'Unimodal'
    :return:
    """
    if fts is None:
        fts = ['Differentiable', 'Separable', 'Unimodal']

    feature_strings = list()
    functions_features = dict()
    for ii in range(len(__all__)):
        function_name = __all__[ii]
        funct = eval("{}(2)".format(function_name))

        feature_str = funct.get_features(fts=fts)
        weight = funct.get_features("string", wrd=wrd, fts=fts)
        functions_features[function_name] = dict(**funct.features, Code=weight)

        feature_strings.append([weight, ii + 1, funct.func_name, feature_str])

    if not rnp:
        # Print first line
        print("Id. & Function Name & Continuous & Differentiable & Separable & Scalable & Unimodal & Convex \\\\")

        # Sort list according to the weight values
        # feature_strings = sorted(feature_strings, key=lambda x: x[0], reverse=True)

        for x in feature_strings:
            print("{} & {} & {} \\\\".format(*x[1:]))
    else:
        return functions_features


def for_all(property, dimension=2):
    if property == 'features':
        return list_functions(rnp=True, fts=None)
    else:
        info = dict()
        # Read all functions and request their optimum data
        for ii in range(len(__all__)):
            function_name = __all__[ii]
            info[function_name] = eval("{}({}).{}".format(function_name, dimension, property))

        return info
