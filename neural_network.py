import tensorflow as tf
import torch
import numpy as np
import pandas as pd
from datasets import Dataset as Dataset_hf
from datasets import load_metric as load_metric_hf 
from os.path import exists as _check_path
from os import makedirs as _create_path
from timeit import default_timer as timer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, DataCollatorWithPadding, DefaultDataCollator, \
        TFAutoModelForSequenceClassification, AutoConfig
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification

def obtain_sample_weight(sample_fitness, fitness_to_weight='rank'):
    """
    Using decreasing functions to give more priority to samples with less fitness

    :param list sample_fitness:
        The fitness associated value for each sample
    
    :param str fitness_to_weight:
        Specify which function use to convert fitness to weight
    
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
        "Pre-process sequences to generate training data for HHNN"
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
        "One-Hot encode the output of the training data"
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
        "Retrieve the pre-processed data"
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
        "Fill a sequence with a dummy value until a fixed length"
        suffix = sequence[:self._memory_length]
        left_len = self._memory_length - len(suffix)
        prefix = [self._num_operators for _ in range(left_len)]
        return prefix + suffix
    
    def __clean_sequence(self, sequence):
        "Keep original values but first -1 element"
        sequence_copy = sequence.copy()
        while len(sequence_copy) > 0 and sequence_copy[0] == -1:
            sequence_copy.pop(0)
        if len(sequence_copy) == 0:
            sequence_copy.append(self._num_operators)  
        return sequence_copy
    
    @staticmethod
    def __lstm_sequence(sequence):
        "Reshape sequence for LSTM architecture usage"
        return [[x] for x in sequence]
    

class ModelPredictorKeras():
    def __init__(self, params):
        # Get encoder
        params['memory_length'] = params.get('memory_length', params['num_steps'])
        self._params = params.copy()
        self._encoder = Encoder(params.copy()).encode
        self.__create_keras_model()
    
    def __create_keras_model(self):
        # Keras TensorFlow Artificial Neural Network Model

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
        num_lstm_layers = sum("LSTM" == layer_type for _, _, layer_type in hidden_layers)
        for idx, (layer_size, layer_activation, layer_type) in enumerate(hidden_layers):
            if layer_type == "Dense":
                self._model.add(tf.keras.layers.Dense(units=layer_size,
                                                      activation=layer_activation))
            elif layer_type == "LSTM":
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
            verbose=False, early_stopping_params=None):

        # Pre-process dataset
        X_encoded = [self._encoder(x) for x in X]
        X_tensor = self.__convert_tensor(X_encoded)
        y_tensor = self.__convert_tensor(y)

        # Callbacks
        callbacks = []
        _, _, filename_dict = retrieve_model_info(self._params)
        # History Logger
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
        callbacks.append(timing_cb)
        
        # Early stopping
        if early_stopping_params is not None:
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
        df_times = pd.DataFrame({"time": timing_cb.logs})
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

class ModelPredictorTransformer():
    def __init__(self, params):
        self._checkpoint = params['pretrained_model']
        self._num_operators = params['num_operators']
        self._params = params.copy()
        self._orig_encoder = Encoder(self._params).encode
        self.__create_tokenizer()
        self.__create_transformer_model()
    
    def __create_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def _encoder(self, sequence):
        sequence_encoded = self._orig_encoder(sequence)
        sequence_str = [' '.join(str(_x) for _x in sequence_encoded)]
        return self._tokenizer(sequence_str)

    def __create_transformer_model(self):

        config = AutoConfig.from_pretrained(
            self._checkpoint,
            vocab_size=len(self._tokenizer),
            n_ctx=1024,
            pad_token_id=self._tokenizer.pad_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            num_labels=self._num_operators
        )
        self._model = AutoModelForSequenceClassification.from_config(config)
        #self._model = AutoModelForSequenceClassification.from_pretrained(self._checkpoint, num_labels=self._num_operators)
    
    def fit(self, X, y, epochs=3, sample_weight=None, 
            verbose=False, early_stopping_params=None):
        _, _, filename_dict = retrieve_model_info(self._params)
        model_directory = filename_dict['model_directory']

        # Prepare dataset
        if sample_weight is None:
            sample_weight = np.ones(len(y)) #/ len(y)
        #else:
        #    sample_weight = 100 * sample_weight / sum(sample_weight)
        y_augmented = [[weight]+y_sample.tolist() for (y_sample, weight) in zip(y, sample_weight.numpy())]
        raw_dataset = Dataset_hf.from_dict({
            'text': [' '.join(str(_x) for _x in x) for x in X],
            'label': y_augmented
        })
        train_dataset = raw_dataset.map(lambda w: self._tokenizer(w['text'], 
                                                                  max_length=1024,
                                                                  truncation=True,
                                                                  padding=True),
                                        batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids',
                                                        'label',
                                                        'attention_mask'])
        
        # Prepare 'accuracy' metric
        #metric = load_metric_hf('accuracy')
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            weights = np.squeeze(labels[..., :1])
            weights = weights / sum(weights)
            reference = np.argmax(labels[..., 1:], axis=-1)
            predictions = np.argmax(logits, axis=-1)
            return {'accuracy': sum(weights * (predictions == reference))}
            #return metric.compute(predictions=predictions, references=reference)
        
        torch.cuda.empty_cache()
        # Training arguments
        batch_size = 32
        training_args = TrainingArguments(
            output_dir=model_directory,
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            eval_steps=1,
            num_train_epochs=epochs, 
            weight_decay=0.01,
            save_strategy='epoch',
            logging_steps=1 if verbose else 500,
            disable_tqdm=not verbose)

        # Integrate sample_weights
        def custom_weighted_loss(labels, logits):
            shift_labels = labels[..., 1:].contiguous()
            weights = torch.squeeze(labels[..., :1].contiguous())
            shift_logits = logits[..., :].contiguous().softmax(dim=1)
            # Categorical Cross Entropy Loss
            loss_val = (-shift_logits.log() * shift_labels).sum(dim=1)
            weighted_loss = (loss_val * weights).mean()
            return weighted_loss
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                input_ids = inputs.get("input_ids")
                labels = inputs.get("labels")
                outputs = model(input_ids)
                loss = custom_weighted_loss(labels, outputs.logits)
                return (loss, outputs) if return_outputs else loss
        
        # Compile model
        data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer, padding=True)
        self._trainer = WeightedTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Fit model
        self._trainer.train()
    
    @staticmethod
    def __softmax(output):
        return np.exp(output) / sum(np.exp(output))
    
    def predict(self, sequence):
        torch.cuda.empty_cache()
        sequence_tokenized = self._encoder(sequence)
        sequence_dataset = Dataset_hf.from_dict(sequence_tokenized)
        prediction, _, _ = self._trainer.predict(sequence_dataset)
        return self.__softmax(prediction)[0]
    
    def load(self, model_path=None):
        _, _, filename_dict = retrieve_model_info(self._params)
        model_directory = filename_dict['model_directory']
        if model_path is None:
            model_path = filename_dict['model_path']
            
        
        if _check_path(model_path):
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self._num_operators)
            training_args = TrainingArguments(output_dir=model_directory, disable_tqdm=True)
            metric = load_metric_hf("accuracy")
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                tokenizer=self._tokenizer,
                compute_metrics=compute_metrics
            )        
            #self._trainer = Trainer(self._model)
            self._predict = self._trainer.predict
            return True
        else:
            raise Exception(f'model_path "{model_path}" does not exists')
    
    def save(self, model_path=None):
        if model_path is None:
            _, _, filename_dict = retrieve_model_info(self._params)
            if not _check_path(filename_dict['model_directory']):
                _create_path(filename_dict['model_directory'])
            model_path = filename_dict['model_path']
        self._trainer.save_model(model_path)



class ModelPredictorKerasTransformer():
    def __init__(self, params):
        self._checkpoint = params['pretrained_model']
        self._num_operators = params['num_operators']
        self._params = params.copy()
        self._orig_encoder = Encoder(self._params).encode
        self.__create_tokenizer()
        self.__create_transformer_model()
    
    def __create_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
    
    def _encoder(self, sequence):
        sequence_encoded = self._orig_encoder(sequence)
        sequence_str = [' '.join(str(_x) for _x in sequence_encoded)]
        return self._tokenizer(sequence_str)

    def __create_transformer_model(self):
        self._model = TFAutoModelForSequenceClassification.from_pretrained(self._checkpoint, num_labels=self._num_operators)
    
    def fit(self, X, y, epochs=3, sample_weight=None, 
            verbose=False, early_stopping_params=None):
        _, _, filename_dict = retrieve_model_info(self._params)
        model_directory = filename_dict['model_directory']

        # Prepare dataset
        if sample_weight is None:
            sample_weight = np.ones(len(y)) #/ len(y)
        #else:
        #    sample_weight = 100 * sample_weight / sum(sample_weight)
        y_augmented = [[weight]+y_sample.tolist() for (y_sample, weight) in zip(y, sample_weight.numpy())]
        raw_dataset = Dataset_hf.from_dict({
            'text': [' '.join(str(_x) for _x in x) for x in X],
            'label': y_augmented
        })
        train_dataset = raw_dataset.map(lambda w: self._tokenizer(w['text']))
        data_collator = DefaultDataCollator(return_tensors="tf", 
                                        batched=True)
        train_dataset = train_dataset.to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            label_cols=["labels"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=8,
        )
        

        # Integrate sample_weights
        def custom_weighted_loss(labels, logits):
            shift_labels = labels[..., 1:].contiguous()
            weights = torch.squeeze(labels[..., :1].contiguous())
            shift_logits = logits[..., :].contiguous().softmax(dim=1)
            # Categorical Cross Entropy Loss
            loss_val = (-shift_logits.log() * shift_labels).sum(dim=1)
            weighted_loss = (loss_val * weights).mean()
            return weighted_loss


        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=custom_weighted_loss,
            metrics=['accuracy']
        )
        
        
        # Callbacks
        callbacks = []
        _, _, filename_dict = retrieve_model_info(self._params)
        # History Logger
        if not _check_path(filename_dict['model_directory']):
            _create_path(filename_dict['model_directory'])
        history_logger = tf.keras.callbacks.CSVLogger(filename_dict['log_path'],
                                                      separator=',', append=True)
        callbacks.append(history_logger)
        # Early stopping
        if early_stopping_params is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_params['monitor'],
                patience=early_stopping_params['patience'],
                mode=early_stopping_params['mode']
            ) 
            callbacks.append(early_stopping)

        self._model.fit(train_dataset,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks)

        # Save predict function        
        self._predict = self._model.predict
    
    @staticmethod
    def __softmax(output):
        return np.exp(output) / sum(np.exp(output))
    
    def predict(self, sequence):
        torch.cuda.empty_cache()
        sequence_tokenized = self._encoder(sequence)
        sequence_dataset = Dataset_hf.from_dict(sequence_tokenized)
        prediction, _, _ = self._predict(sequence_dataset)
        return self.__softmax(prediction)[0]
    
    def load(self, model_path=None):
        _, _, filename_dict = retrieve_model_info(self._params)
        #model_directory = filename_dict['model_directory']
        if model_path is None:
            model_path = filename_dict['model_path']
            
        
        if _check_path(model_path):
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self._num_operators)
            self._trainer = Trainer(model=self._model)        
            self._predict = self._trainer.predict
        else:
            raise Exception(f'model_path "{model_path}" does not exists')
    
    def save(self, model_path=None):
        if model_path is None:
            _, _, filename_dict = retrieve_model_info(self._params)
            if not _check_path(filename_dict['model_directory']):
                _create_path(filename_dict['model_directory'])
            model_path = filename_dict['model_path']
        self._model.save_model(model_path)
        
        
class ModelPredictorTransformerOriginal():
     def __init__(self, params):
         self._checkpoint = params['pretrained_model']
         self._num_operators = params['num_operators']
         self._params = params.copy()
         self._orig_encoder = Encoder(self._params).encode
         self.__create_tokenizer()
         self.__create_transformer_model()

     def __create_tokenizer(self):
         #self._tokenizer = GPT2Tokenizer.from_pretrained(self._params['pretrained_tokenizer'])
         self._tokenizer = AutoTokenizer.from_pretrained(self._params['pretrained_tokenizer'])
         #self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)

     def _encoder(self, sequence):
         sequence_encoded = self._orig_encoder(sequence)
         sequence_str = [', '.join(str(_x) for _x in sequence_encoded)]
         return self._tokenizer(sequence_str)

     def __create_transformer_model(self):
         #self._model = BertForSequenceClassification.from_pretrained(
         #       self._checkpoint,  
         #       num_labels = self._num_operators,
         #       output_attentions = False,
         #       output_hidden_states = False,
         #   )
        #self._model = AutoModelForSequenceClassification.from_pretrained(self._checkpoint, num_labels=self._num_operators)
        config = AutoConfig.from_pretrained(
            self._checkpoint,
            vocab_size=len(self._tokenizer),
            n_ctx=1024,
            pad_token_id=self._tokenizer.pad_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            num_labels=self._num_operators
        )
        self._model = AutoModelForSequenceClassification.from_config(config)
        
        #self._model = GPT2ForSequenceClassification.from_pretrained(self._checkpoint, num_labels=self._num_operators)

        _, _, filename_dict = retrieve_model_info(self._params)
        model_directory = filename_dict['model_directory']             
        training_args = TrainingArguments(output_dir=model_directory, disable_tqdm=True)
        metric = load_metric_hf("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            tokenizer=self._tokenizer,
            compute_metrics=compute_metrics
        )        
        #self._trainer = Trainer(self._model)
        self._predict = self._trainer.predict


     def fit(self, X, y, epochs=3, sample_weight=None, 
             verbose=False, early_stopping_params=None):
         _, _, filename_dict = retrieve_model_info(self._params)
         model_directory = filename_dict['model_directory']

         # Prepare dataset
         raw_dataset = Dataset_hf.from_dict({
             "text": [' '.join(str(_x) for _x in x) for x in X],
             "label": [np.where(y_one==1)[0][0] for y_one in y]
         })
         train_dataset = raw_dataset.map(lambda w: self._tokenizer(w['text']),
                                         batched=True)
         train_dataset.set_format(type='torch', columns=['input_ids',
                                                         'label',
                                                         'attention_mask'])

         # Prepare 'accuracy' metric
         metric = load_metric_hf("accuracy")
         def compute_metrics(eval_pred):
             logits, labels = eval_pred
             predictions = np.argmax(logits, axis=-1)
             return metric.compute(predictions=predictions, references=labels)

         # Training arguments
         batch_size = 32
         torch.cuda.empty_cache()
         training_args = TrainingArguments(
             output_dir=model_directory,
             logging_dir=filename_dict['log_path'],
             evaluation_strategy='epoch',
             learning_rate=3e-5,
             per_device_train_batch_size=batch_size,
             eval_steps=1,
             num_train_epochs=epochs, 
             weight_decay=0.01,
             logging_steps = 1,
             disable_tqdm=not verbose)
         data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer, padding=True)
         self._trainer = Trainer(
             model=self._model,
             args=training_args,
             train_dataset=train_dataset,
             eval_dataset=train_dataset,
             tokenizer=self._tokenizer,
             data_collator=data_collator,
             compute_metrics=compute_metrics
         )

         # Fit model
         self._trainer.train()

         # Save predict function        
         self._predict = self._trainer.predict

     @staticmethod
     def __softmax(output):
         return np.exp(output) / sum(np.exp(output))

     def predict(self, sequence):
        seq_str = ' '.join([str(x) for x in sequence])
        token = self._tokenizer([seq_str], max_length=1024)
        #print(token)
        sequence_dataset = Dataset_hf.from_dict(token)
        #print(sequence_dataset)
        prediction = self._predict(sequence_dataset).predictions[0]
        return np.exp(prediction) / sum(np.exp(prediction))

        # sequence_tokenized = self._encoder(sequence)
        # sequence_dataset = Dataset_hf.from_dict(sequence_tokenized)
        # prediction, _, _ = self._predict(sequence_dataset)
        # return self.__softmax(prediction)[0]

     def load(self, model_path=None):
         _, _, filename_dict = retrieve_model_info(self._params)
         model_directory = filename_dict['model_directory']
         if model_path is None:
             model_path = filename_dict['model_path']


         if _check_path(model_path):
             self._model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self._num_operators)
             training_args = TrainingArguments(output_dir=model_directory, disable_tqdm=True)
             metric = load_metric_hf("accuracy")
             def compute_metrics(eval_pred):
                 logits, labels = eval_pred
                 predictions = np.argmax(logits, axis=-1)
                 return metric.compute(predictions=predictions, references=labels)
             self._trainer = Trainer(
                 model=self._model,
                 args=training_args,
                 tokenizer=self._tokenizer,
                 compute_metrics=compute_metrics
             )        
             #self._trainer = Trainer(self._model)
             self._predict = self._trainer.predict
         else:
             raise Exception(f'model_path "{model_path}" does not exists')

     def save(self, model_path=None):
         if model_path is None:
             _, _, filename_dict = retrieve_model_info(self._params)
             if not _check_path(filename_dict['model_directory']):
                 _create_path(filename_dict['model_directory'])
             model_path = filename_dict['model_path']
         self._trainer.save_model(model_path)

def ModelPredictor(params):
    architecture_name, _, _ = retrieve_model_info(params)
    if architecture_name in ['transformer']:
        return ModelPredictorTransformer(params)
    elif architecture_name in ['transformer_keras']:
        return ModelPredictorKerasTransformer(params)
    elif architecture_name in ['transformer_orig']:
        return ModelPredictorTransformerOriginal(params)
    else:
        return ModelPredictorKeras(params)
