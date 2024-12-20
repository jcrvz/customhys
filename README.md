# customhys

<div>
    <img align="left" alt="Module Dependency Diagram" src="https://raw.githubusercontent.com/jcrvz/customhys/master/docfiles/chm_logo.png" title="Customhys logo" width="25%"/>
</div>
<div align="justify"> 
    <b>Customising optimisation metaheuristics via hyper-heuristic search</b> (CUSTOMHyS). This framework provides tools for solving, but not limited to, continuous optimisation problems using a hyper-heuristic approach for customising metaheuristics. Such an approach is powered by a strategy based on Simulated Annealing. Also, several search operators serve as building blocks for tailoring metaheuristics. They were extracted from ten well-known metaheuristics in the literature.
</div>

Detailed information about this framework can be found in [[1, 2]](#references). Plus, the code for each module is well-documented.


### 🛠 Requirements: 

* Check the [requirements.txt](requirements.txt) file.
* For Apple Silicon, one may need to install TensorFlow via `conda` such as:
```shell
conda install -c apple tensorflow-deps
```
Further information can be found at [Install TensorFlow on Mac M1/M2 with GPU support](https://medium.com/mlearning-ai/install-tensorflow-on-mac-m1-m2-with-gpu-support-c404c6cfb580) by D. Ganzaroli.

## 🧰 Modules

The modules that comprise this framework depend on some basic Python packages, as well as they liaise each other. The module dependency diagram is presented as follows:

![Module Dependency Diagram](https://github.com/jcrvz/customhys/blob/master/docfiles/dependency_diagram.png?raw=true)

**NOTE:** Each module is briefly described below. If you require further information, please check the corresponding source code.

### 🤯 Problems (benchmark functions)

This module includes several benchmark functions as classes to be solved by using optimisation techniques. The class structure is based on Keita Tomochika's repository [optimization-evaluation](https://github.com/keit0222/optimization-evaluation).

Source: [``benchmark_func.py``](customhys/benchmark_func.py)

### 👯‍♂️ Population

This module contains the class Population. A Population object corresponds to a set of agents or individuals within a problem domain. These agents themselves do not explore the function landscape, but they know when to update the position according to a selection procedure.

Source: [``population.py``](customhys/population.py)

### 🦾 Search Operators (low-level heuristics)

This module has a collection of search operators (simple heuristics) extracted from several well-known metaheuristics in the literature. Such operators work over a population, i.e., modify the individuals' positions. 

Source: [``operators.py``](customhys/operators.py)

### 🤖 Metaheuristic (mid-level heuristic)

This module contains the Metaheuristic class. A metaheuristic object implements a set of search operators to guide a population in a search procedure within an optimisation problem.

Source: [``metaheuristic.py``](customhys/metaheuristic.py)

### 👽 Hyper-heuristic (high-level heuristic)

This module contains the Hyperheuristic class. Similar to the Metaheuristic class, but in this case, a collection of search operators is required. A hyper-heuristic object searches within the heuristic space to find the sequence that builds the best metaheuristic for a specific problem.

Source: [``hyperheuristic.py``](customhys/hyperheuristic.py)

### 🏭 Experiment

This module contains the Experiment class.  An experiment object can run several hyper-heuristic procedures for a list of optimisation problems.

Source: [``experiment.py``](customhys/experiment.py)

### 🗜️ Tools

This module contains several functions and methods utilised by many modules in this package.

Source: [``tools.py``](customhys/tools.py)

### 🧠 Machine Learning

This module contains the implementation of Machine Learning models which can power a hyper-heuristic model from this framework. In particular, it is implemented a wrapper for a Neural Network model from Tensorflow. Also, contains auxiliar data structures which process sample of sequences to generate training data for Machine Learning models.

Source: [``machine_learning.py``](customhys/machine_learning.py)

### 💾 Data Structure

The experiments are saved in JSON files. The data structure of a saved file follows a particular scheme described below.

<details>
<summary> Expand structure </summary>
<p>

```
data_frame = {dict: N}
|-- 'problem' = {list: N}
|  |-- 0 = {str}
:  :
|-- 'dimensions' = {list: N}
|  |-- 0 = {int}
:  :
|-- 'results' = {list: N}
|  |-- 0 = {dict: 6}
|  |  |-- 'iteration' = {list: M}   
|  |  |  |-- 0 = {int}
:  :  :  :
|  |  |-- 'time' = {list: M}
|  |  |  |-- 0 = {float}
:  :  :  :
|  |  |-- 'performance' = {list: M}
|  |  |  |-- 0 = {float}
:  :  :  :
|  |  |-- 'encoded_solution' = {list: M}
|  |  |  |-- 0 = {int}
:  :  :  :
|  |  |-- 'solution' = {list: M}
|  |  |  |-- 0 = {list: C}
|  |  |  |  |-- 0 = {list: 3}
|  |  |  |  |  |-- search_operator_structure
:  :  :  :  :  :
|  |  |-- 'details' = {list: M}
|  |  |  |-- 0 = {dict: 4}
|  |  |  |  |-- 'fitness' = {list: R}
|  |  |  |  |  |-- 0 = {float}
:  :  :  :  :  :
|  |  |  |  |-- 'positions' = {list: R}
|  |  |  |  |  |-- 0 = {list: D}
|  |  |  |  |  |  |-- 0 = {float}
:  :  :  :  :  :  :
|  |  |  |  |-- 'historical' = {list: R}
|  |  |  |  |  |-- 0 = {dict: 5}
|  |  |  |  |  |  |-- 'fitness' = {list: I}
|  |  |  |  |  |  |  |-- 0 = {float}
:  :  :  :  :  :  :  :
|  |  |  |  |  |  |-- 'positions' = {list: I}
|  |  |  |  |  |  |  |-- 0 = {list: D}
|  |  |  |  |  |  |  |  |-- 0 = {float}
:  :  :  :  :  :  :  :  :
|  |  |  |  |  |  |-- 'centroid' = {list: I}
|  |  |  |  |  |  |  |-- 0 = {list: D}
|  |  |  |  |  |  |  |  |-- 0 = {float}
:  :  :  :  :  :  :  :  :
|  |  |  |  |  |  |-- 'radius' = {list: I}
|  |  |  |  |  |  |  |-- 0 = {float}
:  :  :  :  :  :  :  :
|  |  |  |  |  |  |-- 'stagnation' = {list: I}
|  |  |  |  |  |  |  |-- 0 = {int}
:  :  :  :  :  :  :  :
|  |  |  |  |-- 'statistics' = {dict: 10}
|  |  |  |  |  |-- 'nob' = {int}
|  |  |  |  |  |-- 'Min' = {float}
|  |  |  |  |  |-- 'Max' = {float}
|  |  |  |  |  |-- 'Avg' = {float}
|  |  |  |  |  |-- 'Std' = {float}
|  |  |  |  |  |-- 'Skw' = {float}
|  |  |  |  |  |-- 'Kur' = {float}
|  |  |  |  |  |-- 'IQR' = {float}
|  |  |  |  |  |-- 'Med' = {float}
|  |  |  |  |  |-- 'MAD' = {float}
:  :  :  :  :  :
```
where:
- ```N``` is the number of files within data_files folder
- ```M``` is the number of hyper-heuristic iterations (metaheuristic candidates)
- ```C``` is the number of search operators in the metaheuristic (cardinality)
- ```P``` is the number of control parameters for each search operator
- ```R``` is the number of repetitions performed for each metaheuristic candidate
- ```D``` is the dimensionality of the problem tackled by the metaheuristic candidate
- ```I``` is the number of iterations performed by the metaheuristic candidate
- ```search_operator_structure``` corresponds to ```[operator_name = {str}, control_parameters = {dict: P}, selector = {str}]```
</p>
</details>

## 🏗️ Work-in-Progress

The following modules are available, but they may do not work. They are currently under developing.

### 🌡️ Characterisation

This module intends to provide metrics for characterising the benchmark functions.

Source: [``characterisation.py``](customhys/characterisation.py)

### 📊 Visualisation

This module intends to provide several tools for plotting results from the experiments.

Source: [``visualisation.py``](customhys/visualisation.py)

## Sponsors

<a href="https://tec.mx/en" target="_blank"><img src="https://github.com/jcrvz/customhys/raw/master/docfiles/logoTEC_full.png" width="250"></a>
<a href="http://www.cas.cn/" target="_blank"><img src="https://github.com/jcrvz/customhys/raw/master/docfiles/cas_logo.png" width="250"></a>
<a href="https://www.gob.mx/conacyt" target="_blank"><img src="https://github.com/jcrvz/customhys/raw/master/docfiles/conacyt-logo.png" width="250"></a>

## References

### Seminal Papers
The seminal papers that describe the framework's theoretical background and software implementation are:
1. [J. M. Cruz-Duarte, I. Amaya, J. C. Ortiz-Bayliss, H. Terashima-Marín, and Y. Shi, _CUSTOMHyS: Customising Optimisation Metaheuristics via Hyper-heuristic Search_, **SoftwareX**, vol. 12, p. 100628, 2020.](https://www.sciencedirect.com/science/article/pii/S2352711020303411)
1. [J. M. Cruz-Duarte, J. C. Ortiz-Bayliss, I. Amaya, Y. Shi, H. Terashima-Marín, and N. Pillay, _Towards a Generalised Metaheuristic Model for Continuous Optimisation Problems_, **Mathematics**, vol. 8, no. 11, p. 2046, Nov. 2020.](https://www.mdpi.com/2227-7390/8/11/2046)
1. [J. M. Cruz-Duarte, I. Amaya, J. C. Ortiz-Bayliss, S. E. Connat-Pablos, and H. Terashima-Marín, _A Primary Study on Hyper-Heuristics to Customise Metaheuristics for Continuous Optimisation_. **2020 IEEE Congress on Evolutionary Computation (CEC)**, 2020.](docfiles/SearchOperators_CEC.pdf)
1. [J. M. Cruz-Duarte, I. Amaya, J. C. Ortiz-Bayliss, S. E. Conant-Pablos, H. Terashima-Marín, H., and Y. Shi. _Hyper-Heuristics to Customise Metaheuristics for Continuous Optimisation_, **Swarm and Evolutionary Computation**, 100935.](https://doi.org/10.1016/j.swevo.2021.100935)

### Published Journal Papers
These are the journal articles that have been published using this framework:
1. [J. M. Tapia-Avitia, J. M. Cruz‐Duarte, I. Amaya, J. C. Ortiz-Bayliss, H. Terashima-Marín, and N. Pillay, _Analysing Hyper-Heuristics based on Neural Networks for the Automatic Design of Population-based Metaheuristics in Continuous Optimisation Problems_, **Swarm and Evolutionary Computation**, 89, 101616, 2024.](https://doi.org/10.1016/j.swevo.2024.101616)
1. [D. F. Zambrano-Gutierrez, G. H. Valencia-Rivera, J. G. Avina-Cervantes, I. Amaya, and J. M. Cruz-Duarte, _Designing Heuristic-Based Tuners for Fractional-Order PID Controllers in Automatic Voltage Regulator Systems Using a Hyper-heuristic Approach_, **Fractal Fract**, 2024.](https://doi.org/10.3390/fractalfract8040223)
1. [D. F. Zambrano-Gutierrez, J. M. Cruz-Duarte, J. G. Avina-Cervantes, J. C. Ortiz-Bayliss, J. J. Yanez-Borjas, and I. Amaya, _Automatic Design of Metaheuristics for Practical Engineering Applications_, **IEEE Access**., vol. 11, pp. 7262-7276, 2023.](https://doi.org/10.1109/ACCESS.2023.3236836)
1. [J. M. Cruz-Duarte, J. C. Ortiz-Bayliss, I. Amaya, and N. Pillay, _Global Optimisation through Hyper-Heuristics: Unfolding Population-Based Metaheuristics_, **Appl. Sci.**, vol. 11, no. 12, p. 5620, 2021.](http://dx.doi.org/10.3390/app11125620)

### Presented Conference Papers
These are the conference articles that have been presented using this framework:
1. [D. F.  Zambrano-Gutierrez, J. M. Cruz-Duarte, J. C. Ortiz-Bayliss, I. Amaya, and J. G. Avina-Cervantes, _Beyond Traditional Tuning: Unveiling Metaheuristic Operator Trends in PID Control Tuning for Automatic Voltage Regulation_, **2024 IEEE Congress on Evolutionary Computation (CEC)**, 2024.](https://doi.org/10.1109/CEC60901.2024.10611968)
1. [G. Pérez-Espinosa, J. M. Cruz-Duarte, I. Amaya, J. C. Ortiz-Bayliss, H. Terashima-Marín, and N. Pillay, _Tailoring Metaheuristics for Designing Thermodynamic-Optimal Cooling Devices for Microelectronic Thermal Management Applications_, **2024 IEEE Congress on Evolutionary Computation (CEC)**, 2024.](https://doi.org/10.1109/CEC60901.2024.10611938)
1. [D. Acosta-Ugalde, J. M. Cruz-Duarte, S. E. Conant-Pablos, and J. G. Falcón-Cardona, _Beyond 'Novel' Metaphor-based Metaheuristics: An Interactive Algorithm Design Software_, **2024 IEEE Congress on Evolutionary Computation (CEC)**, 2024.](https://doi.org/10.1109/CEC60901.2024.10611912)
1. [D. F. Zambrano-Gutierrez, A. C. Molina-Porras, J. G. Avina-Cervantes, R. Correa, and J. M. Cruz-Duarte, _Designing Heuristic-Based Tuners for PID Controllers in Automatic Voltage Regulator Systems Using an Automated Hyper-Heuristic Approach_, **2023 IEEE Symposium Series on Computational Intelligence (SSCI)**, Mexico City, Mexico, 2023, pp. 1263-1268.](https://doi.org/10.1109/SSCI52147.2023.10371925)
1. [D. F. Zambrano-Gutierrez, A. C. Molina-Porras, E. Ovalle-Magallanes, I. Amaya, J. C. Ortiz-Bayliss, J. G. Avina-Cervantes, and J. M. Cruz-Duarte,  _SIGNRL: A Population-Based Reinforcement Learning Method for Continuous Control_, **2023 IEEE Symposium Series on Computational Intelligence (SSCI)**, Mexico City, Mexico, 2023, pp. 1443-1448.](https://doi.org/10.1109/SSCI52147.2023.10371875)
1. [D. F. Zambrano-Gutierrez, J. M. Cruz-Duarte, and H. Castañeda, _Automatic Hyper-Heuristic to Generate Heuristic-based Adaptive Sliding Mode Controller Tuners for Buck-Boost Converters_, in **The Genetic and Evolutionary Computation Conference (GECCO)**, 2023, pp. 1-8. **Nominated to Best Paper Award**](https://dl.acm.org/doi/10.1145/3583131.3590510)
1. [J. M. Tapia-Avitia, J. M. Cruz-Duarte, I. Amaya, J. C. Ortiz-Bayliss, H. Terashima-Marin, and N. Pillay. _A Primary Study on Hyper-Heuristics Powered by Artificial Neural Networks for Customising Population-based Metaheuristics in Continuous Optimisation Problems_, **2022 IEEE Congress on Evolutionary Computation (CEC)**, 2022.](https://doi.org/10.1109/CEC55065.2022.9870275)
1. [J. M. Cruz-Duarte, I. Amaya, J. C. Ortiz-Bayliss, N. Pillay. _A Transfer Learning Hyper-heuristic Approach for Automatic Tailoring of Unfolded Population-based Metaheuristics_, **2022 IEEE Congress on Evolutionary Computation (CEC)**, 2022.](https://doi.org/10.1109/CEC55065.2022.9870426)
1. [J. M. Cruz-Duarte, I. Amaya, J. C. Ortiz-Bayliss, N. Pillay. _Automated Design of Unfolded Metaheuristics and the Effect of Population Size_. **2021 IEEE Congress on Evolutionary Computation (CEC)**, 1155–1162, 2021.](https://doi.org/10.1109/CEC45853.2021.9504879)