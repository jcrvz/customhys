# -*- coding: utf-8 -*-
"""
This module contains Machine Learning tools.

Created on Wed Sep  8 00:00:00 2021

@author: Jose Manuel Tapia Avitia, e-mail: josetapia@exatec.tec.mx
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import exists as _check_path
from os import makedirs as _create_path
from timeit import default_timer as timer


def obtain_sample_weight(sample_fitness, fitness_to_weight='rank'):
    """
    Using decreasing functions to give more priority to samples with less fitness
    :param list sample_fitness: The fitness associated value for each sample
    :param str fitness_to_weight: Specify which function use to convert fitness to weight
    :return: An array that associates a weight to each sample
    """
    a = min(sample_fitness)
    b = max(sample_fitness)
    if fitness_to_weight == 'linear_reciprocal':
        # f: [a, b] -> (0, 1]
        weight_conversion = lambda fitness: a / fitness
    elif fitness_to_weight == 'linear_reciprocal_translated':
        # f: [a, b] -> [1, b-a+1]
        weight_conversion = lambda fitness: a * b / fitness - a + 1
    elif fitness_to_weight == 'linear_percentage':
        # f: [a, b] -> [1, 100]
        weight_conversion = lambda fitness: 100 * (b - fitness) / (b - a) + 1
    elif fitness_to_weight == 'rank':
        # f: [fitness] -> [0, n-1]
        indices = list(range(len(sample_fitness)))
        indices.sort(key = lambda idx: -sample_fitness[idx])
        indices_sorted = [0 for _ in indices]
        for i, idx in enumerate(indices):
            indices_sorted[idx] = i
        return indices_sorted
    else:
        # Default linear conversion
        # f: [a, b] -> [a, b]
        weight_conversion = lambda fitness: a + b - fitness
    
    return [weight_conversion(fitness) for fitness in sample_fitness]


class DatasetSequences():
    def __init__(self, sequences, fitnesses, num_operators=None, fitness_to_weight=None):
        'Pre-process sequences to generate training data for HHNN'
        X, y, sample_fitness = [], [], []
        for sequence, fitness in zip(sequences, fitnesses):
            if len(sequence) >0 and sequence[0] == -1:
                sequence.pop(0)
                fitness.pop(0)
            while len(sequence) > 0:
                # Per each prefix, predict the next operator
                y.append(sequence.pop())
                X.append(sequence.copy())
                sample_fitness.append(fitness.pop())            
        self._X = X
        self._y = y
        if fitness_to_weight is not None:
            self._sample_weight =\
                tf.constant(obtain_sample_weight(sample_fitness, fitness_to_weight))
        else:
            self._sample_weight = None
        self._one_hot_encoded = None
        if num_operators is not None:
            self.apply_one_hot_encoding(num_operators)
    
    def apply_one_hot_encoding(self, num_operators):
        'One-Hot encode the output of the training data'
        if self._one_hot_encoded is not None:
            if self._one_hot_encoded != num_operators:
                self._y = [np.where(y_one==1)[0][0] for y_one in self._y]
            else:
                return
        self._one_hot_encoded = num_operators
        if num_operators is not None:
            self._y = tf.one_hot(indices=self._y,
                                 depth=num_operators,
                                 dtype=tf.int64).numpy()
    
    def obtain_dataset(self):
        'Retrieve the pre-processed data'
        return self._X, self._y, self._sample_weight


def retrieve_model_info(params):
    # Check essential attributes
    essential_attributes = ['file_label', 'model_architecture', 'encoder', 'num_steps']
    if not all(attribute in params for attribute in essential_attributes):
        left_attributes = [attribute for attribute in essential_attributes if attribute not in params]
        raise Exception(f'The following attributes left while retrieving the model info: {left_attributes}')
    
    # Names
    architecture_name = params['model_architecture']
    encoder_name = params['encoder']

    # Model label
    memory_length = params.get('memory_length', params['num_steps'])
    attribute_labels = [params['file_label'], f'mem{memory_length}']
    if 'pretrained_model' in params:
        attribute_labels.append(params['pretrained_model'])
    model_label = '-'.join(attribute_labels)
    personal_label = params['file_label'].split('_')
    if personal_label[-1] == 'extended':
        personal_label.pop()
    personal_labels = '_'.join(personal_label)

    # Filenames
    model_directory = './data_files/ml_models/'
    model_filename = model_label
    if architecture_name == 'transformer':
        model_directory = model_directory +f'{model_label}_dir/'
        model_filename = 'trained_model'

    filename_dict = dict({
        'model_directory': model_directory,
        'model_label': model_label,
        'model_path': model_directory + f'{personal_labels}.h5',
        'log_path': model_directory + f'{model_filename}_log.csv',
        'log_time_path': model_directory + f'{model_filename}_log_time.csv'
    })
    
    return architecture_name, encoder_name, filename_dict


class Encoder():
    def __init__(self, params):
        self._encoder_name = params['encoder']
        self._architecture_name = params['model_architecture']
        self._memory_length = params.get('memory_length', 100)
        self._num_operators = params['num_operators']

        def one_hot_encode(sequence):
            return tf.one_hot(indices=sequence, depth=self._num_operators,
                              dtype=tf.int64).numpy()
        def compose(f, g):
            return lambda x: f(g(x))
        
        # Choice identity encoder
        if self._architecture_name in ['transformer', 'transformer_orig', 'LSTM_Ragged']:
            # Keep original values but element -1
            self.__identity_encoder = self.__clean_sequence
        else:
            # Keep original values but fix the length
            self.__identity_encoder = compose(self.__fix_sequence_length, self.__clean_sequence)

        # Get encoder module
        if self._encoder_name in ['identity', 'default']:
            encoder = self.__identity_encoder
        elif self._encoder_name in ['one_hot_encoder']:
            # Fix sequence, then one-hot encode it
            encoder = compose(one_hot_encode, self.__identity_encoder)
        else:
            raise Exception('Encoder name does not exists')
        
        # Prepare if LSTM is used
        if self._architecture_name in ['LSTM']:
            # Encode sequence, then reshape
            self._encoder = compose(self.__lstm_sequence, encoder)
        else:
            self._encoder = encoder
    
    def encode(self, sequence):
        return self._encoder(sequence)
    
    def __fix_sequence_length(self, sequence):
        'Fill a sequence with a dummy value until a fixed length'
        suffix = sequence[:self._memory_length]
        left_len = self._memory_length - len(suffix)
        prefix = [self._num_operators for _ in range(left_len)]
        return prefix + suffix
    
    def __clean_sequence(self, sequence):
        'Keep original values but first -1 element'
        sequence_copy = sequence.copy()
        while len(sequence_copy) > 0 and sequence_copy[0] == -1:
            sequence_copy.pop(0)
        if len(sequence_copy) == 0:
            sequence_copy.append(self._num_operators)  
        return sequence_copy
    
    @staticmethod
    def __lstm_sequence(sequence):
        'Reshape sequence for LSTM architecture usage'
        return [[x] for x in sequence]
    

class ModelPredictorKeras():
    # Keras TensorFlow Artificial Neural Network Model
    def __init__(self, params):
        # Get encoder
        params['memory_length'] = params.get('memory_length', params['num_steps'])
        self._params = params.copy()
        self._encoder = Encoder(params.copy()).encode
        self.__create_keras_model()
    
    def __create_keras_model(self):
        # Create model
        self._model = tf.keras.Sequential()
        self._num_operators = self._params['num_operators']
        architecture_name = self._params['model_architecture']
        input_size = self._params['memory_length']
        hidden_layers = self._params['model_architecture_layers']
        
        # Input layer
        if architecture_name in ['MLP']:
            # MLP input
            self._model.add(tf.keras.Input(shape=input_size))
        elif architecture_name in ['LSTM_Ragged']:
            # Variable length, supported using ragged tensors
            max_length_sequence = self._params['num_steps']
            self._model.add(tf.keras.Input(shape=(max_length_sequence,)))

            # Embedding layer
            if len(hidden_layers) > 0:
                first_layer_size, _, _ = hidden_layers[0]
            else:
                first_layer_size = self._num_operators
            self._model.add(tf.keras.layers.Embedding(self._num_operators + 1, first_layer_size))
        elif architecture_name in ['LSTM']:
            # LSTM input
            self._model.add(tf.keras.Input(shape=(input_size, 1)))

        # Hidden layers
        num_lstm_layers = sum('LSTM' == layer_type for _, _, layer_type in hidden_layers)
        for idx, (layer_size, layer_activation, layer_type) in enumerate(hidden_layers):
            if layer_type == 'Dense':
                self._model.add(tf.keras.layers.Dense(units=layer_size,
                                                      activation=layer_activation))
            elif layer_type == 'LSTM':
                self._model.add(tf.keras.layers.LSTM(units=layer_size,
                                                     activation=layer_activation,
                                                     return_sequences=idx + 1 < num_lstm_layers))        

        # Output layer
        self._model.add(tf.keras.layers.Dense(self._num_operators, activation='softmax'))
    
        # Compile model
        self._model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(),
                            metrics=['accuracy'])
    
    def __convert_tensor(self, tensor):
        if self._params['model_architecture'] in ['LSTM_Ragged']:
            return tf.ragged.constant(tensor)
        else:
            return tf.constant(tensor)
        
    def fit(self, X, y, epochs=100, sample_weight=None, 
            verbose=False, early_stopping_params=None, verbose_statistics=False):

        # Pre-process dataset
        X_encoded = [self._encoder(x) for x in X]
        X_tensor = self.__convert_tensor(X_encoded)
        y_tensor = self.__convert_tensor(y)

        # Callbacks
        callbacks = []
        _, _, filename_dict = retrieve_model_info(self._params)

        # History Logger
        if verbose_statistics:
            if not _check_path(filename_dict['model_directory']):
                _create_path(filename_dict['model_directory'])
            history_logger = tf.keras.callbacks.CSVLogger(filename_dict['log_path'],
                                                        separator=',', append=True)
            callbacks.append(history_logger)

        class TimingCallback(tf.keras.callbacks.Callback):
            def __init__(self, logs={}):
                self.logs = []
            def on_epoch_begin(self, epoch, logs={}):
                self.start_time = timer()
            def on_epoch_end(self, epoch, logs={}):
                self.logs.append(timer() - self.start_time)
        timing_cb = TimingCallback()
        if verbose_statistics:
            callbacks.append(timing_cb)
        
        # Early stopping
        if early_stopping_params is not None and verbose_statistics:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_params['monitor'],
                patience=early_stopping_params['patience'],
                mode=early_stopping_params['mode']
            ) 
            callbacks.append(early_stopping)

        # Train model
        self._model.fit(X_tensor, y_tensor, 
                  epochs=epochs,
                  sample_weight=sample_weight, 
                  verbose=verbose,
                  callbacks=callbacks)
        if verbose_statistics:
            df_times = pd.DataFrame({'time': timing_cb.logs})
            df_times.to_csv(filename_dict['log_time_path'])
            
        # Save predict function
        self._predict = self._model.predict

    def predict(self, sequence):
        # Use model to predict weights
        tensor = self.__convert_tensor([self._encoder(sequence)])
        return self._predict(tensor)[0]
    
    def load(self, model_path=None):
        if model_path is None:
            _, _, filename_dict = retrieve_model_info(self._params)
            model_path = filename_dict['model_path']
            
        if _check_path(model_path):
            self._model = tf.keras.models.load_model(model_path)    
            # Save predict function
            self._predict = self._model.predict
            return True
        else:
            raise Exception(f'model_path "{model_path}" does not exists')

    def save(self, model_path=None):
        if model_path is None:
            _, _, filename_dict = retrieve_model_info(self._params)
            if not _check_path(filename_dict['model_directory']):
                _create_path(filename_dict['model_directory'])
            model_path = filename_dict['model_path']
        self._model.save(model_path)


def ModelPredictor(params):
    # Function that decide which ML model uses.
    # For now, it is only supported the NN architecture with dense 
    # and lstm layers. Future version will support Transformers.
    return ModelPredictorKeras(params)
