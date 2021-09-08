from metaheuristic import Metaheuristic
import hyperheuristic as hh
import numpy as np
import os
import tensorflow as tf
import operators as Operators

class HyperheuristicML(hh.Hyperheuristic):
    def solve(self, mode='neural_network', kw_parameters={}):
        if mode == 'neural_network':
            return self._solve_neural_network(kw_parameters)
        else:
            return super().solve(mode, kw_parameters)
    
    def _solve_neural_network(self, kw_nn_params):
        # Reproducible
        tf.random.set_seed(1)
        np.random.seed(1)

        sequence_per_repetition = list()
        fitness_per_repetition = list()

        for rep_model in range(1, kw_nn_params['num_models']+1):
            # Neural network
            params = dict(load_model=kw_nn_params['load_model'],
                        sequences_size=kw_nn_params['sequences_size'], 
                        model_path=kw_nn_params['model_path'],
                        kw_weighting_params=kw_nn_params['kw_weighting_params'],
                        save_model=kw_nn_params['save_model'])
            model = self.get_neural_network_model(params);

            for rep in range(1, kw_nn_params['num_replicas']+1):
                # Metaheuristic
                mh = Metaheuristic(self.problem, num_agents=self.parameters['num_agents'], 
                                    num_iterations=self.num_iterations)

                # Initialiser
                mh.apply_initialiser()

                # Extract the population and fitness values, and their best values
                current_fitness = np.copy(mh.pop.global_best_fitness)
                current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

                # Heuristic sets
                self.current_space = np.arange(self.num_operators)

                # Initialise additional variables
                candidate_enc_so = list()
                current_sequence = [-1]
                
                best_fitness = [current_fitness]
                best_position = [current_position]

                step = 0
                stag_counter = 0
                exclude_idx = []

                # Finalisator
                while step < kw_nn_params['sequences_size'] and stag_counter <= self.parameters['stagnation_percentage']*kw_nn_params['sequences_size']:
                    # Use model to predict weights
                    sequence_input = self.encodeSequence(current_sequence.copy(), kw_nn_params['sequences_size'])
                    weights_output = model.predict(tf.constant([sequence_input]))[0]
                    for idx in exclude_idx:
                        weights_output[idx] = 0
                    if sum(weights_output)>0:
                        weights_output /= sum(weights_output)
                    else:
                        weights_output = np.ones(self.num_operators)/self.num_operators

                    # Pick a simple heuristic and apply it
                    candidate_enc_so = self._obtain_candidate_solution(sol=1, operators_weights=weights_output)
                    candidate_search_operator = self.get_operators([candidate_enc_so[-1]])
                    perturbators, selectors = Operators.process_operators(candidate_search_operator)

                    mh.apply_search_operator(perturbators[0], selectors[0])

                    # Extract population and fitness values
                    current_fitness = np.copy(mh.pop.global_best_fitness)
                    current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

                    # Print update
                    if self.parameters['verbose']:
                        print(
                            '{} :: Model:{:3d}, Rep: {:3d}, Step: {:3d}, Trial: {:3d}, SO: {:30s}, currPerf: {:.2e}, candPerf: {:.2e}, '
                            'csl: {:3d}'.format(
                                self.file_label, rep_model, rep, step, stag_counter,
                                candidate_search_operator[0][0] + ' & ' + candidate_search_operator[0][2][:4],
                                best_fitness[-1], current_fitness, len(self.current_space)), end=' ')

                    # If the candidate solution is better or equal than the current best solution
                    if current_fitness < best_fitness[-1]:
                        # Update the current sequence and its characteristics
                        current_sequence.append(candidate_enc_so[-1])

                        best_fitness.append(current_fitness)
                        best_position.append(current_position)

                        # Update counters
                        step += 1
                        stag_counter = 0
                        exclude_idx = []

                        # Add improvement mark
                        if self.parameters['verbose']:
                            print('+', end='')

                    else:  # Then try another search operator
                        # Revert the modification to the population in the mh object
                        mh.pop.revert_positions()

                        # Update stagnation
                        stag_counter += 1
                        if stag_counter % kw_nn_params['delete_idx'] == 0:
                            exclude_idx.append(candidate_enc_so[-1])

                    # Add ending mark
                    if self.parameters['verbose']:
                        print('')

                # Print the best one
                if self.parameters['verbose']:
                    print('\nBest fitness: {},\nBest position: {}'.format(current_fitness, current_position))

                #  Update the repetition register
                sequence_per_repetition.append(np.double(current_sequence).astype(int).tolist())
                fitness_per_repetition.append(np.double(best_fitness).tolist())

        return fitness_per_repetition, sequence_per_repetition

    
    
    def encodeSequence(self, seq, sequences_size):        
        if seq and seq[0] == -1:
            seq.pop(0)
        
        flatten = lambda arr: [item for list_arr in arr for item in list_arr] 

        seqPadded = np.pad(seq, 
                        (0, sequences_size-len(seq)), 
                        'constant', 
                        constant_values=(-1,))

        return flatten(tf.one_hot(indices=seqPadded, depth=self.num_operators).numpy())

    def get_neural_network_model(self, kw_model_params={}):
        """
            Params:
                - load_model : Boolean that says if we want to load or not a model
                - sequencesSize : The maximum lenght admisible for a dynamic sequence
                - model_path : Directoy where the model will be saved
                - kw_weighting_params : Params for _solve_dynamic
                - save_model : Boolean that says if we want to save or not the model
        """

        model_directory = './ml_models/'
        model_path = os.path.join(model_directory, kw_model_params['model_path'])
        if kw_model_params['load_model'] and os.path.isdir(model_path):
            return tf.keras.models.load_model(model_path)



        # Data generation
        _, seqrep, _, _ = self._solve_dynamic(kw_model_params['kw_weighting_params'])

        X, y = [], []
        for seq in seqrep:
            if seq and seq[0] == -1:
                seq.pop(0)
            while seq:
                y.append(tf.one_hot(indices=[seq.pop()], depth=self.num_operators).numpy()[0])
                X.append(self.encodeSequence(seq, kw_model_params['sequences_size']))
        X, y = tf.constant(X), tf.constant(y)

        # Model

        # Create the model
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(kw_model_params['sequences_size']*self.num_operators)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(self.num_operators, activation='softmax')
        ])
        
        # Compile the model
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])

        # Fit the model
        model.fit(X, y, epochs = 100)

        # Save model
        if kw_model_params['save_model']:
            if not os.path.isdir(model_directory):
                os.mkdir(model_directory)
            model.save(model_path)

        return model
        
if __name__ == '__main__':
    import benchmark_func as bf
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as st
    from pprint import pprint
    plt.rcParams.update({'font.size': 18,
                         "text.usetex": True,
                         "font.family": "serif"})

    problem = bf.Sphere(50)
    # problem = bf.Stochastic(50)
    # problem = bf.CosineMixture(50)
    # problem = bf.Whitley(50)
    # problem = bf.Schwefel220(50)
    # problem = bf.Sargan(45)

    # problem = bf.choose_problem('<random>', np.random.randint(2, 50))
    # problem.set_search_range(-10, 10)

    file_label = "{}-{}D".format(problem.func_name, problem.variable_num)

    q = HyperheuristicML(problem=problem.get_formatted_problem(),
                       heuristic_space='short_collection.txt',  # 'default.txt',  #
                       file_label=file_label)
    q.parameters['num_agents'] = 30
    q.parameters['num_steps'] = 100
    q.parameters['stagnation_percentage'] = 0.5
    q.parameters['num_replicas'] = 200
    sampling_portion = 0.37  # 0.37

    num_models_nn = 10
    num_replicas_nn = 10
    fitprep_nn, seqrep_nn = q.solve('neural_network', {
        'num_models':num_models_nn,
        'num_replicas':num_replicas_nn, 
        'sequences_size':q.parameters['num_steps'],
        'delete_idx':4,
        'load_model':False,
        'save_model':True,
        'model_path':'model_nn',
        'kw_weighting_params': {
            'include_fitness': False,
            'learning_portion': sampling_portion
        }})

    q.parameters['num_replicas'] = 100
    #sampling_portion = 0.37  # 0.37
    fitprep_dyn, seqrep_dyn, _, _ = q.solve('dynamic', {
            'include_fitness': False,
            'learning_portion': sampling_portion
        })

    #fitprep = np.concatenate((fitprep_dyn, fitprep_nn))
    fitprep = fitprep_dyn.copy()
    for seq in fitprep_nn:
        fitprep.append(seq)
    #seqrep = np.concatenate((seqrep_dyn, seqrep_nn))
    seqrep = seqrep_dyn.copy()
    for seq in seqrep_nn:
        seqrep.append(seq)
    colours = plt.cm.rainbow(np.linspace(0, 1, len(fitprep)))

    folder_name = './figures-to-export/'

    # ------- Figure 1
    fi1 = plt.figure(figsize=(8, 3))
    plt.ion()
    for x, c in zip(fitprep, colours):
        plt.plot(x, '-o', color=c)
    plt.xlabel('Step')
    plt.ylabel('Fitness')
    plt.ioff()
    # plt.plot(c, 'o')
    plt.savefig(folder_name + file_label + "_FitnesStep" + ".svg", dpi=333, transparent=True)
    fi1.show()

    # ------- Figure 2
    fi2 = plt.figure(figsize=(6, 6))
    ax = fi2.add_subplot(111, projection='3d')
    plt.ion()
    for x, y, c in zip(fitprep, seqrep, colours):
        ax.plot3D(range(1, 1 + len(x)), y, x, 'o-', color=c)

    plt.xlabel('Step')
    plt.ylabel('Search Operator')
    ax.set_zlabel('Fitness')
    plt.ioff()
    plt.savefig(folder_name + file_label + "_SOStepFitness" + ".svg", dpi=333, transparent=True)
    fi2.show()

    # ------- Figure 3
    new_colours = plt.cm.jet(np.linspace(0, 1, len(fitprep)))

    # ------- Figure 4
    """
    if weimatrix is not None:
        # plt.figure()
        fi3 = plt.figure(figsize=(8, 3))
        plt.imshow(weimatrix.T, cmap="hot_r")
        plt.xlabel('Step')
        plt.ylabel('Search Operator')
        plt.savefig(folder_name + file_label + "_SOStep" + ".svg", dpi=333, transparent=True)
        fi3.show()
    """

    # ------- Figure 5
    get_last_fitness = lambda fitlist: np.array([ff[-1] for ff in fitlist])
    last_fitness_values = get_last_fitness(fitprep)
    last_fitness_values_nn = get_last_fitness(fitprep_nn)
    last_fitness_values_dyn = get_last_fitness(fitprep_dyn)
    midpoint = int(q.parameters['num_replicas'] * sampling_portion)
    

    fi4 = plt.figure(figsize=(8, 3))
    plt.boxplot([last_fitness_values_nn, last_fitness_values_dyn[:midpoint], last_fitness_values_dyn[midpoint:], last_fitness_values],
                showmeans=True)
    plt.xticks(range(1, 5), ['Neural network', 'Train', 'Test/Refine', 'All'])
    plt.ylabel('Fitness')
    plt.xlabel('Sample')
    plt.savefig(folder_name + file_label + "FitnessSample" + ".svg", dpi=333, transparent=True)
    plt.show()

    # print('Stats for all fitness values:')
    pprint(st.describe(last_fitness_values[:midpoint])._asdict())
