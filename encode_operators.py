from operators import get_operator_aliases

def get_operator_param_names():
  return {
  'random_search': ['scale', 'distribution'],
  'central_force_dynamic': ['gravity', 'alpha', 'beta', 'dt'],
  'differential_mutation': ['expression', 'num_rands', 'factor'],
  'firefly_dynamic': ['distribution', 'alpha', 'beta', 'gamma'],
  'genetic_crossover': ['pairing', 'crossover', 'mating_pool_factor'],
  'genetic_mutation': ['scale', 'elite_rate', 'mutation_rate', 'distribution'],
  'gravitational_search': ['gravity', 'alpha'],
  'random_flight': ['scale', 'distribution'],
  'local_random_walk': ['probability', 'scale', 'distribution'],
  'random_sample': [],
  'spiral_dynamic': ['radius', 'angle', 'sigma'],
  'swarm_dynamic': ['factor', 'self_conf', 'swarm_conf', 'version', 'distribution']
  }

def get_inverse_operator_aliases():
  return {
  'RS': 'random_search',
  'CF': 'central_force_dynamic',
  'DM': 'differential_mutation',
  'FD': 'firefly_dynamic',
  'GC': 'genetic_crossover',
  'GM': 'genetic_mutation',
  'GS': 'gravitational_search',
  'RF': 'random_flight',
  'RW': 'local_random_walk',
  'RX': 'random_sample',
  'SD': 'spiral_dynamic',
  'PS': 'swarm_dynamic'
  }, {
    'g': 'greedy', 
    'd': 'all', 
    'm': 'metropolis', 
    'p': 'probabilistic'
  }


def get_inverse_str_args():
  return {
  'u': 'uniform',
  'rand': 'rand',
  'best': 'best',
  'curr': 'current',
  'currtb': 'current-to-best',
  'rtb': 'rand-to-best',
  'rtbc': 'rand-to-best-and-current',
  'g': 'gaussian',
  'levy': 'levy',
  'rank': 'rank',
  's': 'single',
  'cost': 'cost',
  'r': 'random',
  't': 'tournament_2_100',
  'two': 'two',
  'b': 'blend',
  'l': 'linear_0.5_0.5',
  'i': 'inertial',
  'c': 'constriction'
  }
  


def convert_param(param_value):
  if type(param_value) == str:
    if param_value == 'rand-to-best-and-current':
      return 'rtbc', 'rand-to-best-and-current'
    if param_value == 'rand-to-best':
      return 'rtb', 'rand-to-best'
    if param_value == 'current-to-best':
      return 'currtb', 'current-to-best'
    if param_value == 'current':
      return 'curr', 'current'
    param_components = param_value.split('-')
    component_compressed = []
    for component in param_components:
      if len(component) > 4 and component != 'current':
        component_compressed.append(component[0])
      else:
        component_compressed.append(component)
    str_components = ''.join(component_compressed)
    return str_components, param_value
  else:
    return str(param_value), param_value
  

def get_inverse_arg(arg_value):
  try:
    # Return float number if possible
    return float(arg_value)
  except:
    # Decompress string
    return get_inverse_str_args()[arg_value]

  
def compress_operator(operator):
  perturbator_alias, selector_alias = get_operator_aliases()
  perturbator, args, selector = operator
  params = ';'.join([convert_param(arg)[0] for arg in args.values()])
  conv_perturbator = perturbator_alias[perturbator]
  conv_selector = selector_alias[selector]
  compress_operator = ','.join([conv_perturbator, params, conv_selector])
  pad_sz = 25 - len(compress_operator)
  return compress_operator + '_' * pad_sz



def decompress_operator(compress_operator):
  conv_perturbator, params, conv_selector = compress_operator.replace('_', '').split(',')
  inverse_perturbator_alias, inverse_selector_alias = get_inverse_operator_aliases()
  perturbator = inverse_perturbator_alias[conv_perturbator]
  selector = inverse_selector_alias[conv_selector]
  compressed_arg_values = params.split(';')
  arg_values = [get_inverse_arg(arg_value) for arg_value in compressed_arg_values]
  args = {}
  arg_names = get_operator_param_names()[perturbator]
  for key, value in zip(arg_names, arg_values):
    args[key] = value
  return perturbator, args, selector
  