# %%
from hyperheuristic import Hyperheuristic
import benchmark_func as bf
import numpy as np
from metaheuristic import Metaheuristic
import operators as Operators
import os
import benchmark_func as bf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from pprint import pprint
import tikzplotlib as ptx
from sklearn.preprocessing import normalize
import seaborn as sns


# %%
params_seqs = dict({
  "filters": {
    "features": None,
    "include_dimensions": True,
    "include_population": True,
    "sequence_limit": 100
  },
  "retrieve_sequences": True,
  "generate_sequences": False,
  "store_sequences": False,
  "kw_weighting_params": {
    "include_fitness": False,
    "learning_portion": 1
  }
})

# Reproducible for test purposes
np.random.seed(1)
problem = bf.Sphere(45)
file_label = "{}-{}D".format(problem.func_name, problem.variable_num)

q = Hyperheuristic(problem=problem.get_formatted_problem(),
                    heuristic_space='default.txt',  # 'default.txt',  #short_collection  automatic medium_collection
                    file_label=file_label)

# %%
# Obtain training data
seqfitness, seqrep = q._get_sample_sequences(params_seqs) 
X, y, sample_fitness = q._process_sample_sequences(seqfitness, seqrep)







# %%

training_dataset = []

max_len = max(len(x) for x in X)
for idx, (x, _y) in enumerate(zip(X, y)):
  original_len = len(x)
  missing_len = max_len - original_len
  
  x = np.pad(x, (0, missing_len), constant_values=0)
  token_types = np.zeros(max_len)
  attention = [1 for _ in range(original_len)] + [0 for _ in range(missing_len)]
  
  
  training_dataset.append(dict({
    "input_ids": x,
    "labels": _y.index(1),
    "token_type_ids": token_types,
    "attention_mask": attention,
    #"idx": idx
  }))





# %%  
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from datasets import load_metric
from transformers import Trainer
from datasets import Dataset
from transformers import DataCollatorWithPadding

# %%
checkpoint = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=205)
# ignore_mismatched_sizes=True
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# %%
training_args = TrainingArguments(output_dir="test_trainer_distiled_lr05")
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

# %%
X = [[0], [1, 2], [4, 3, 2, 4]]
y = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
X_str = [' '.join(str(_x) for _x in x) for x in X]

dataset = Dataset.from_dict({
  "text": X_str,
  "label": [_y.index(1) for _y in y]
})
tokenized_dataset = dataset.map(preprocess_function, batched=True)
columns_to_return = ['input_ids', 'label', 'attention_mask']
tokenized_dataset.set_format(type='torch', columns=columns_to_return)

#%%
training_args = TrainingArguments(
    output_dir="./results_distiled_new",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# %%


# Load test data
X_test = [[], [54], [54, 196], [54, 196, 196], [54, 196, 196, 196], [54, 196, 196, 196, 196], [54, 196, 196, 196, 196, 100], [54, 196, 196, 196, 196, 100, 99]]
# %%
X_str_test = [' '.join(str(_x) for _x in x) for x in X_test]
X_test_tokenized = tokenizer(X_str_test, padding=True, truncation=True, max_length=512)


#dataset_test = Dataset.from_dict({
#  "text": X_str_test
#})
#tokenized_dataset = dataset.map(preprocess_function, batched=True)

#columns_to_return = ['input_ids', 'label', 'attention_mask']
#tokenized_dataset.set_format(type='torch', columns=columns_to_return)



# Create torch dataset
test_dataset = Dataset.from_dict(X_test_tokenized)

# Load trained model
model_path = "results_distiled/checkpoint-3000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=205)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
y_pred
X_test.append(y_pred)
y_pred

# %%

def softmax_n(A):
  return np.exp(A) / sum(np.exp(A))

# %%

def run_little_exp(model_trained):
  q.parameters['num_steps']=100
  sequence_per_repetition = list()
  fitness_per_repetition = list()
  weights_per_repetition = list()
  for rep in range(100):
    
    mh = Metaheuristic(q.problem, num_agents=q.parameters['num_agents'], num_iterations=q.num_iterations)

    # Initialiser
    mh.apply_initialiser()

    # Extract the population and fitness values, and their best values
    current_fitness = np.copy(mh.pop.global_best_fitness)
    current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

    current_space = np.arange(q.num_operators)

    # Initialise additional variables
    candidate_enc_so = list()
    current_sequence = [-1]

    best_fitness = [current_fitness]
    best_position = [current_position]

    step = 0
    stag_counter = 0
    exclude_indices = []

    # Finalisator
    while not q._check_finalisation(step, stag_counter):
      # Use the trained model to predict operators weights
      
      X_test = [current_sequence[1:]]
      X_str_test = [' '.join(str(_x) for _x in x) for x in X_test]
      X_test_tokenized = tokenizer(X_str_test, padding=True, truncation=True, max_length=512)
      test_dataset = Dataset.from_dict(X_test_tokenized)
      raw_pred, _, _ = model_trained.predict(test_dataset, )
      operators_weights = softmax(raw_pred[0])

      # Select a simple heuristic and apply it
      candidate_enc_so = q._obtain_candidate_solution(sol=1, operators_weights=operators_weights)
      candidate_search_operator = q.get_operators([candidate_enc_so[-1]])
      perturbators, selectors = Operators.process_operators(candidate_search_operator)

      mh.apply_search_operator(perturbators[0], selectors[0])

      # Extract population and fitness values
      current_fitness = np.copy(mh.pop.global_best_fitness)
      current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

      # Print update
      print(
          '{} :: Neural Network, Rep: {:3d}, Step: {:3d}, Trial: {:3d}, SO: {:30s}, currPerf: {:.2e}, candPerf: {:.2e}, '
          'csl: {:3d}'.format(
              q.file_label, rep + 1, step + 1, stag_counter,
              candidate_search_operator[0][0] + ' & ' + candidate_search_operator[0][2][:4],
              best_fitness[-1], current_fitness, len(q.current_space)), end=' ')

      # If the candidate solution is better or equal than the current best solution
      if current_fitness < best_fitness[-1]:
          # Update the current sequence and its characteristics
          current_sequence.append(candidate_enc_so[-1])

          best_fitness.append(current_fitness)
          best_position.append(current_position)

          # Update counters
          step += 1
          stag_counter = 0
          # Reset tabu list
          exclude_indices = []

          # Add improvement mark
          print('+', end='')

      else:  # Then try another search operator
          # Revert the modification to the population in the mh object
          mh.pop.revert_positions()

          # Update stagnation
          stag_counter += 1
          if stag_counter % 5 == 0:
              # Include last search operator's index to the tabu list
              exclude_indices.append(candidate_enc_so[-1])

      # Add ending mark
      print('')

    # Print the best one
    print('\nBest fitness: {},\nBest position: {}'.format(current_fitness, current_position))

    # Update the repetition register
    sequence_per_repetition.append(np.double(current_sequence).astype(int).tolist())
    fitness_per_repetition.append(np.double(best_fitness).tolist())
  return fitness_per_repetition, sequence_per_repetition

# %%


fitseq, seqrep = run_little_exp(test_trainer)
fitprep_nn = fitseq
seqrep_nn = seqrep

q.parameters["num_replicas"] = 100
sampling_portion = 0.37
fitprep_dyn, seqrep_dyn, _ = q.solve('dynamic', {
    'include_fitness': False,
    'learning_portion': sampling_portion
})

fitprep = fitprep_dyn.copy()
for seq in fitprep_nn:
    fitprep.append(seq)
seqrep = seqrep_dyn.copy()
for seq in seqrep_nn:
    seqrep.append(seq)
    
    

colours = plt.cm.rainbow(np.linspace(0, 1, len(fitprep)))

# is there a way to update the weight matrix using the information provided from each run

# ------- Figure 0
fi0 = plt.figure()
plt.ion()

# Find max length
max_length = max([x.__len__() for x in seqrep])
mat_seq = np.array([np.array([*x, *[-2] * (max_length - len(x))]) for x in seqrep], dtype=object).T

bins = np.arange(-2, 30 + 1)
current_hist = list()
for step in range(max_length):
    dummy_hist = np.histogram(mat_seq[step, :], bins=bins, density=True)[0][2:]
    current_hist.append(dummy_hist)

sns.heatmap(np.array(current_hist).T, linewidths=.5, cmap='hot_r')

plt.xlabel('Step')
# plt.yticks(range(30, step=2), range(start=1, stop=31, step=2))
plt.ylabel('Operator')
plt.ioff()
# plt.plot(c, 'o')
plt.show()


# ------- Figure 1
fi1 = plt.figure(figsize=(8, 3))
plt.ion()
for x, c in zip(fitprep, colours):
    plt.plot(x, '-o', color=c)
plt.xlabel('Step')
plt.ylabel('Fitness')
plt.ioff()
# plt.plot(c, 'o')
# plt.savefig(folder_name + file_label + "_FitnesStep" + ".svg", dpi=333, transparent=True)
fi1.show()

# Figure 2

fi2 = plt.figure(figsize=(6, 6))
ax = fi2.add_subplot(111, projection='3d')
plt.ion()
for x, y, c in zip(fitprep, seqrep, colours):
    ax.plot3D(range(1, 1 + len(x)), y, x, 'o-', color=c)

plt.xlabel('Step')
plt.ylabel('Search Operator')
ax.set_zlabel('Fitness')
plt.ioff()
#plt.savefig(folder_name + file_label + "_SOStepFitness" + ".svg", dpi=333, transparent=True)
fi2.show()
    
# Figure 3

get_last_fitness = lambda fitlist: np.array([ff[-1] for ff in fitlist])
last_fitness_values = get_last_fitness(fitprep)
last_fitness_values_nn = get_last_fitness(fitprep_nn)
last_fitness_values_dyn = get_last_fitness(fitprep_dyn)
midpoint = int(q.parameters['num_replicas'] * sampling_portion)

fi4 = plt.figure(figsize=(8, 3))
plt.boxplot([last_fitness_values_nn, last_fitness_values_dyn[:midpoint], last_fitness_values_dyn[midpoint:],
              last_fitness_values],
            showmeans=True)
plt.xticks(range(1, 5), ['Neural network', 'Train', 'Test/Refine', 'All'])

plt.ylabel('Fitness')
plt.xlabel('Sample')
#plt.savefig(folder_name + file_label + "FitnessSample" + ".svg", dpi=333, transparent=True)
plt.show()

pprint(st.describe(last_fitness_values_nn)._asdict())
pprint(st.describe(last_fitness_values_dyn)._asdict())
pprint(st.describe(last_fitness_values)._asdict())