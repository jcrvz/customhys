
# coding: utf-8

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__all__ =  ['Ackley','Sphere','Rosenbrock','Beale','GoldsteinPrice','Booth',
            'BukinN6','Matyas','LeviN13','ThreeHumpCamel','Easom','Eggholder',
            'McCormick','SchafferN2','SchafferN4','StyblinskiTang','DeJongsF1',
            'DeJongsF2','DeJongsF3','DeJongsF4','DeJongsF5','Ellipsoid','KTablet',
            'FiveWellPotential','WeightedSphere','HyperEllipsodic',
            'SumOfDifferentPower','Griewank','Michalewicz','Perm','Rastrigin',
            'Schwefel','SixHumpCamel','Shuberts','XinSheYang','Zakharov']

__oneArgument__ = ['Beale','GoldsteinPrice','Booth','BukinN6','Matyas','LeviN13',
                   'ThreeHumpCamel','Easom','Eggholder','McCormick','SchafferN2',
                   'SchafferN4','DeJongsF3','DeJongsF4','DeJongsF5',
                   'FiveWellPotential','SixHumpCamel','Shuberts']

__twoArgument__ = ['Ackley','Sphere','Rosenbrock','StyblinskiTang','DeJongsF1',
                   'DeJongsF2','Ellipsoid','KTablet','WeightedSphere',
                   'HyperEllipsodic','SumOfDifferentPower','Griewank',
                   'Michalewicz','Rastrigin','Schwefel','XinSheYang','Zakharov']

__threeArgument__ = ['Perm']

##### Basic function #####
class OptimalBasic:
    def __init__(self, variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([0]*self.variable_num)
        self.min_search_range = np.array([0]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.plot_place = 0.25
        self.func_name = ''
        self.save_dir = os.path.dirname(os.path.abspath(__file__))+'\\img\\'
        if(os.path.isdir(self.save_dir) == False):
            os.mkdir(self.save_dir)
    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        return -1

    def plot(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], self.plot_place, dtype=np.float32)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], self.plot_place, dtype=np.float32)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[2:self.variable_num]))
                z.append(self.get_func_val(np.array(tmp)))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

    def save_fig(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], self.plot_place, dtype=np.float32)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], self.plot_place, dtype=np.float32)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[2:self.variable_num]))
                z.append(self.get_func_val(np.array(tmp)))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.savefig(self.save_dir+self.func_name+'.png')
        plt.close()

##### Optimization benchmark function group #####
##### Class Ackley function #####
class Ackley(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([32.768]*self.variable_num)
        self.min_search_range = np.array([-32.768]*self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.func_name = 'Ackley'

    def get_func_val(self, variables):
        tmp1 = 20.-20.*np.exp(-0.2*np.sqrt(1./self.variable_num*np.sum(np.square(variables))))
        tmp2 = np.e-np.exp(1./self.variable_num*np.sum(np.cos(variables*2.*np.pi)))
        return tmp1+tmp2

##### Class Sphere function #####
class Sphere(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1000]*self.variable_num) # nearly inf
        self.min_search_range = np.array([-1000]*self.variable_num) # nearly inf
        self.optimal_solution = np.array([0]*self.variable_num)
        self.global_optimum_solution = 0
        self.plot_place = 10
        self.func_name = 'Sphere'

    def get_func_val(self, variables):
        return np.sum(np.square(variables))

##### Class Rosenbrock function #####
class Rosenbrock(OptimalBasic):
    def __init__(self, variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5]*self.variable_num)
        self.min_search_range = np.array([-5]*self.variable_num)
        self.optimal_solution = np.array([1]*self.variable_num)
        self.global_optimum_solution = 0
        self.plot_place = 0.25
        self.func_name = 'Rosenbrock'

    def get_func_val(self, variables):
        f = 0
        for i in range(self.variable_num-1):
            f += 100*np.power(variables[i+1]-np.power(variables[i],2),2)+np.power(variables[i]-1,2)
        return f

##### Class Beale function #####
class Beale(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([4.5]*self.variable_num)
        self.min_search_range = np.array([-4.5]*self.variable_num)
        self.optimal_solution = np.array([3.,0.5])
        self.global_optimum_solution = 0
        self.plot_place = 0.25
        self.func_name = 'Beale'

    def get_func_val(self, variables):
        tmp1 = np.power(1.5 - variables[0] + variables[0] * variables[1],2)
        tmp2 = np.power(2.25 - variables[0] + variables[0] * np.power(variables[1],2),2)
        tmp3 = np.power(2.625 - variables[0] + variables[0] * np.power(variables[1],3),2)
        return tmp1+tmp2+tmp3

##### Class Goldstein-Price function #####
class GoldsteinPrice(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([2.]*self.variable_num)
        self.min_search_range = np.array([-2.]*self.variable_num)
        self.optimal_solution = np.array([0.,-1.])
        self.global_optimum_solution = 3
        self.plot_place = 0.25
        self.func_name = 'GoldsteinPrice'

    def get_func_val(self, variables):
        tmp1 = (1+np.power(variables[0]+variables[1]+1,2)*(19-14*variables[0]+3*np.power(variables[0],2)-14*variables[1]+6*variables[0]*variables[1]+3*np.power(variables[1],2)))
        tmp2 = (30+(np.power(2*variables[0]-3*variables[1],2)*(18-32*variables[0]+12*np.power(variables[0],2)+48*variables[1]-36*variables[0]*variables[1]+27*np.power(variables[1],2))))
        return tmp1*tmp2

##### Class Booth function #####
class Booth(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([1.,3.])
        self.global_optimum_solution = 0
        self.func_name = 'Booth'

    def get_func_val(self, variables):
        tmp1 = np.power(variables[0]+2*variables[1]-7,2)
        tmp2 = np.power(2*variables[0]+variables[1]-5,2)
        return tmp1+tmp2

##### Class Bukin function N.6 #####
class BukinN6(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([-5.,3.])
        self.min_search_range = np.array([-15.,-3.])
        self.optimal_solution = np.array([-10.,1.])
        self.global_optimum_solution = 0
        self.func_name = 'BukinN6'

    def get_func_val(self, variables):
        tmp1 = 100*np.sqrt(np.absolute(variables[1]-0.01*np.power(variables[1],2)))
        tmp2 = 0.01*np.absolute(variables[0]+10)
        return tmp1+tmp2

##### Class Matyas function #####
class Matyas(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([0.,0.])
        self.global_optimum_solution = 0
        self.func_name = 'Matyas'

    def get_func_val(self, variables):
        tmp1 = 0.26*(np.power(variables[0],2)+np.power(variables[1],2))
        tmp2 = 0.48*variables[0]*variables[1]
        return tmp1-tmp2

##### Class Levi function N.13 #####
class LeviN13(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([10.]*self.variable_num)
        self.min_search_range = np.array([-10.]*self.variable_num)
        self.optimal_solution = np.array([1.,1.])
        self.global_optimum_solution = 0
        self.func_name = 'LeviN13'

    def get_func_val(self, variables):
        tmp1 = np.power(np.sin(3*np.pi*variables[0]),2)
        tmp2 = np.power(variables[0]-1,2)*(1+np.power(np.sin(3*np.pi*variables[1]),2))
        tmp3 = np.power(variables[1]-1,2)*(1+np.power(np.sin(2*np.pi*variables[1]),2))
        return tmp1+tmp2+tmp3

##### Class Three-hump camel function #####
class ThreeHumpCamel(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([5.]*self.variable_num)
        self.min_search_range = np.array([-5.]*self.variable_num)
        self.optimal_solution = np.array([0.,0.])
        self.global_optimum_solution = 0
        self.func_name = 'ThreeHumpCamel'

    def get_func_val(self, variables):
        return 2*np.power(variables[0],2)-1.05*np.power(variables[0],4)+np.power(variables[0],6)/6+variables[0]*variables[1]+np.power(variables[1],2)

##### Class Easom function #####
class Easom(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100.]*self.variable_num)
        self.optimal_solution = np.array([np.pi,np.pi])
        self.global_optimum_solution = -1
        self.plot_place = 10
        self.func_name = 'Easom'

    def get_func_val(self, variables):
        return -1.0*np.cos(variables[0])*np.cos(variables[1])*np.exp(-(np.power(variables[0]-np.pi,2)+np.power(variables[1]-np.pi,2)))

##### Class Eggholder function #####
class Eggholder(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([512.]*self.variable_num)
        self.min_search_range = np.array([-512.]*self.variable_num)
        self.optimal_solution = np.array([512.,404.2319])
        self.global_optimum_solution = -959.6407
        self.plot_place = 5
        self.func_name = 'Eggholder'

    def get_func_val(self, variables):
        tmp1 = -(variables[1]+47)*np.sin(np.sqrt(np.absolute(variables[1]+variables[0]/2+47)))
        tmp2 = -variables[0]*np.sin(np.sqrt(np.absolute(variables[0]-(variables[1]+47))))
        return tmp1+tmp2

##### Class McCormick function #####
class McCormick(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([4.]*self.variable_num)
        self.min_search_range = np.array([-1.5,-3.])
        self.optimal_solution = np.array([-0.54719,-1.54719])
        self.global_optimum_solution = -1.9133
        self.func_name = 'McCormick'

    def get_func_val(self, variables):
        tmp1 = np.sin(variables[0]+variables[1])+np.power(variables[0]-variables[1],2)
        tmp2 = -1.5*variables[0]+2.5*variables[1]+1
        return tmp1+tmp2

##### Class Schaffer function N.2 #####
class SchafferN2(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100]*self.variable_num)
        self.optimal_solution = np.array([0.,0.])
        self.global_optimum_solution = 0
        self.plot_place = 10
        self.func_name = 'SchafferN2'

    def get_func_val(self, variables):
        tmp1 = np.power(np.sin(np.power(variables[0],2)-np.power(variables[1],2)),2)-0.5
        tmp2 = np.power(1+0.001*(np.power(variables[0],2)+np.power(variables[1],2)),2)
        return 0.5+tmp1/tmp2

##### Class Schaffer function N.4 #####
class SchafferN4(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([100.]*self.variable_num)
        self.min_search_range = np.array([-100]*self.variable_num)
        self.optimal_solution = np.array([0.,1.253115])
        self.global_optimum_solution = 0.292579
        self.plot_place = 10
        self.func_name = 'SchafferN4'

    def get_func_val(self, variables):
        tmp1 = np.power(np.cos(np.sin(np.absolute(np.power(variables[0],2)-np.power(variables[1],2)))),2)-0.5
        tmp2 = np.power(1+0.001*(np.power(variables[0],2)+np.power(variables[1],2)),2)
        return 0.5+tmp1/tmp2

##### Class Styblinski-Tang function #####
class StyblinskiTang(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.]*self.variable_num)
        self.min_search_range = np.array([-5.]*self.variable_num)
        self.optimal_solution = np.array([-2.903534]*self.variable_num)
        self.global_optimum_solution = -39.166165*self.variable_num
        self.func_name = 'StyblinskiTang'

    def get_func_val(self, variables):
        tmp1 = 0
        for i in range(self.variable_num):
        	tmp1 += np.power(variables[i],4)-16*np.power(variables[i],2)+5*variables[i]
        return tmp1/2

##### Class De Jong's function F1 #####
class DeJongsF1(Sphere):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.func_name = 'DeJongsF1'

##### Class De Jong's function F2 #####
class DeJongsF2(Rosenbrock):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.func_name = 'DeJongsF2'

##### Class De Jong's function F3 #####
class DeJongsF3(OptimalBasic):
    def __init__(self):
        super().__init__(5)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([-5.12]*self.variable_num)
        self.global_optimum_solution = 0
        self.func_name = 'DeJongsF3'

    def get_func_val(self, variables):
        tmp1 = 0
        for i in range(self.variable_num):
        	tmp1 += np.floor(variables[i])
        return tmp1

##### Class De Jong's function F4 #####
class DeJongsF4(OptimalBasic):
    def __init__(self):
        super().__init__(30)
        self.max_search_range = np.array([1.28]*self.variable_num)
        self.min_search_range = np.array([-1.28]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = np.random.normal(0,1)
        self.func_name = 'DeJongsF4'

    def get_func_val(self, variables):
        tmp1 = 0
        for i in range(self.variable_num):
        	tmp1 += (i+1)*np.power(variables[i],4)
        return tmp1 + np.random.normal(0, 1)

##### Class De Jong's function F5 #####
class DeJongsF5(OptimalBasic):
    def __init__(self):
        super().__init__(25)
        self.max_search_range = np.array([65.536]*self.variable_num)
        self.min_search_range = np.array([-65.536]*self.variable_num)
        self.optimal_solution = np.array([-32.32]*self.variable_num)
        self.global_optimum_solution = 1.
        self.plot_place = 1.5
        self.func_name = 'DeJongsF5'

    def get_func_val(self, variables):
        A = np.zeros([2,25])
        a = [-32,16,0,16,32]
        A[0,:] = np.tile(a,(1,5))
        tmp = []
        for x in a:
            tmp_list = [x]*5
            tmp.extend(tmp_list)
        A[1,:] = tmp

        sum = 0
        for i in range(self.variable_num):
            a1i = A[0,i]
            a2i = A[1,i]
            term1 = i
            term2 = np.power(variables[0]-a1i,6)
            term3 = np.power(variables[1]-a2i,6)
            new = 1/(term1+term2+term3)
            sum += new
        return 1/(0.002+sum)

##### Class Ellipsoid function #####
class Ellipsoid(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Ellipsoid'

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += np.power(np.power(1000,i/(self.variable_num-1))*variables[i],2)
        return tmp

##### Class k-tablet function #####
class KTablet(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'KTablet'

    def get_func_val(self, variables):
        tmp = 0
        k = int(self.variable_num/4)
        for i in range(k):
            tmp += variables[i]

        for i in range(k,self.variable_num):
            tmp += np.power(100*variables[i],2)
        return tmp

##### Class Five-well potential function #####
# Not yet checked to do working properly
class FiveWellPotential(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([20.]*self.variable_num)
        self.min_search_range = np.array([-20.]*self.variable_num)
        self.optimal_solution = np.array([4.92,-9.89])
        self.global_optimum_solution = -1.4616
        self.plot_place = 1
        self.func_name = 'FiveWellPotential'

    def get_func_val(self, variables):
        tmp1 = []
        tmp1.append(1-1/(1+0.05*np.power(np.power(variables[0],2)+(variables[1]-10),2)))
        tmp1.append(-1/(1+0.05*(np.power(variables[0]-10,2)+np.power(variables[1],2))))
        tmp1.append(-1/(1+0.03*(np.power(variables[0]+10,2)+np.power(variables[1],2))))
        tmp1.append(-1/(1+0.05*(np.power(variables[0]-5,2)+np.power(variables[1]+10,2))))
        tmp1.append(-1/(1+0.1*(np.power(variables[0]+5,2)+np.power(variables[1]+10,2))))
        tmp1_sum = 0
        for x in tmp1:
            tmp1_sum += x
        tmp2 = 1+0.0001*np.power((np.power(variables[0],2)+np.power(variables[1],2)),1.2)
        return tmp1_sum*tmp2

##### Class Weighted Sphere function or hyper ellipsodic function #####
class WeightedSphere(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'WeightedSphere'

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += (i+1)*np.power(variables[i],2)
        return tmp

class HyperEllipsodic(WeightedSphere):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.func_name = 'HyperEllipsodic'

##### Class Sum of different power function #####
class SumOfDifferentPower(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1.]*self.variable_num)
        self.min_search_range = np.array([-1.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'SumOfDifferentPower'

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += np.power(np.absolute(variables[i]),i+2)
        return tmp

##### Class Griewank function #####
class Griewank(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([600.]*self.variable_num)
        self.min_search_range = np.array([-600.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Griewank'

    def get_func_val(self, variables):
        tmp1 = 0
        tmp2 = 1
        for i in range(self.variable_num):
            tmp1 += np.power(variables[i],2)
            tmp2 = tmp2*np.cos(variables[i]/np.sqrt(i+1))
        return tmp1/4000-tmp2

##### Class Michalewicz function #####
class Michalewicz(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([np.pi]*self.variable_num)
        self.min_search_range = np.array([0.]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = -1.8013 # In case of variable_num == 2
        self.plot_place = 0.1
        self.func_name = 'Michalewicz'

    def get_func_val(self, variables):
        m = 10
        tmp1 = 0
        for i in range(self.variable_num):
            tmp1 += np.sin(variables[i])*np.power(np.sin((i+1)*np.power(variables[i],2)/np.pi),2*m)
        return -tmp1

##### Class Perm function #####
class Perm(OptimalBasic):
    def __init__(self,variable_num,beta):
        super().__init__(variable_num)
        self.beta = beta
        self.max_search_range = np.array([1.]*self.variable_num)
        self.min_search_range = np.array([-1.]*self.variable_num)
        tmp = []
        for i in range(self.variable_num):
            tmp.append(1/(i+1))
        self.optimal_solution = np.array(tmp)
        self.global_optimum_solution = 0.
        self.plot_place = 0.1
        self.func_name = 'Perm'

    def get_func_val(self, variables):
        tmp1 = 0
        tmp2 = 0
        for j in range(self.variable_num):
            for i in range(self.variable_num):
                tmp1 += (i+1+self.beta)*(np.power(variables[i],j+1)-np.power(1/(i+1),j+1))
            tmp2 += np.power(tmp1,2)
            tmp1 = 0
        return tmp2

##### Class Rastrigin function #####
class Rastrigin(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([5.12]*self.variable_num)
        self.min_search_range = np.array([-5.12]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Rastrigin'

    def get_func_val(self, variables):
        tmp1 = 10 * self.variable_num
        tmp2 = 0
        for i in range(self.variable_num):
            tmp2 += np.power(variables[i],2)-10*np.cos(2*np.pi*variables[i])
        return tmp1+tmp2

##### Class Schwefel function #####
class Schwefel(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([500.]*self.variable_num)
        self.min_search_range = np.array([-500.]*self.variable_num)
        self.optimal_solution = np.array([420.9687]*self.variable_num)
        self.global_optimum_solution = -418.9829
        self.plot_place = 10.
        self.func_name = 'Schwefel'

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += variables[i]*np.sin(np.sqrt(np.absolute(variables[i])))
        return -tmp

##### Class Six-hump camel function #####
class SixHumpCamel(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([3.,2.])
        self.min_search_range = np.array([-3.,-2.])
        self.optimal_solution = np.array([-0.0898,0.7126])
        self.global_optimum_solution = -1.0316
        self.func_name = 'SixHumpCamel'

    def get_func_val(self, variables):
        return 4-2.1*np.power(variables[0],2)+1/3*np.power(variables[0],4)*np.power(variables[0],2)+variables[0]*variables[1]+4*(np.power(variables[1],2)-1)*np.power(variables[1],2)

##### Class Shuberts function #####
class Shuberts(OptimalBasic):
    def __init__(self):
        super().__init__(2)
        self.max_search_range = np.array([1000.,10.]) # Set infinite as 1000 for x1
        self.min_search_range = np.array([-10.,-1000]) # Set infinite as -1000 for x2
        self.optimal_solution = np.array([0.,0.])
        self.global_optimum_solution = -186.7309
        self.plot_place = 10.
        self.func_name = 'Shuberts'

    def get_func_val(self, variables):
        n = 5
        tmp1 = 0
        tmp2 = 0
        for i in range(n):
            tmp1 += (i+1)*np.cos((i+1)+(i+2)*variables[0])
            tmp2 += (i+1)*np.cos((i+1)+(i+2)*variables[1])
        return tmp1*tmp2

##### Class Xin-She Yang function #####
class XinSheYang(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([2.*np.pi]*self.variable_num)
        self.min_search_range = np.array([-2.*np.pi]*self.variable_num)
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'XinSheYang'

    def get_func_val(self, variables):
        tmp1 = 0
        tmp2 = 0
        for i in range(self.variable_num):
            tmp1 += np.absolute(variables[i])
            tmp2 += np.sin(np.power(variables[i],2))
        return tmp1*np.exp(-tmp2)

##### Class Zakharov function #####
class Zakharov(OptimalBasic):
    def __init__(self,variable_num):
        super().__init__(variable_num)
        self.max_search_range = np.array([1000.]*self.variable_num) # temporarily set as 1000
        self.min_search_range = np.array([-1000]*self.variable_num) # temporarily set as -1000
        self.optimal_solution = np.array([0.]*self.variable_num)
        self.global_optimum_solution = 0.
        self.plot_place = 10.
        self.func_name = 'Zakharov'

    def get_func_val(self, variables):
        tmp1 = 0
        tmp2 = 0
        for i in range(self.variable_num):
            tmp1 += variables[i]
            tmp2 += (i+1)*variables[i]
        return tmp1+np.power(1/2*tmp2,2)+np.power(1/2*tmp2,4)
