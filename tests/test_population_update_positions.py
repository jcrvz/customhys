import pytest
import numpy as np
from customhys import benchmark_func as bf
from customhys import population as pp
selector_data = {'all', 'greedy'}
selector_global_data = {'greedy'}
selectors = {('all', 'greedy'),
             ('greedy', 'all'),
             ('greedy', 'greedy'),
             ('all', 'all')
             }

@pytest.fixture
def pop():
    fun = bf.Sphere(2)
    population = pp.Population(fun.get_search_range(), num_agents=2)

    population.initialise_positions('vertex')
    population.evaluate_fitness(lambda x: fun.get_function_value(x))

    return population

@pytest.fixture
def pop_mod():
    fun = bf.Sphere(2)
    population = pp.Population(fun.get_search_range(), num_agents=2)

    # Initialise the population
    np.random.seed(69)
    population.initialise_positions('random')
    population.evaluate_fitness(lambda x: fun.get_function_value(x))

    print("\nInitial fitness values")
    print(population.fitness, population.previous_fitness, population.backup_fitness)
    population.update_positions(level='population', selector='all')
    population.update_positions(level='particular', selector='all')
    population.update_positions(level='global', selector='greedy')
    print(population.fitness, population.previous_fitness, population.backup_fitness)
    print("\n")

    population.previous_global_best_fitness = population.global_best_fitness

    # Perform a perturbation of the population
    new_positions = population.positions
    new_positions[0] = new_positions[0] / 2
    new_positions[-1] = new_positions[-1] * 2
    population.positions = new_positions
    population.evaluate_fitness(lambda x: fun.get_function_value(x))

    return population

@pytest.mark.parametrize("selector", selector_data)
def test_update_initial_positions_by_population(pop, selector):
    pop.update_positions(level='population', selector=selector)

    assert not np.array_equal(pop.positions, pop.previous_positions, equal_nan=True)

@pytest.mark.parametrize("selector", selector_data)
def test_update_initial_positions_by_particular(pop, selector):
    pop.update_positions(level='population', selector='all')
    pop.update_positions(level='particular', selector=selector)

    assert np.array_equal(pop.positions, pop.particular_best_positions, equal_nan=True)

@pytest.mark.parametrize("selector", selector_global_data)
def test_update_initial_positions_by_global(pop, selector):
    pop.update_positions(level='population', selector='all')
    pop.update_positions(level='global', selector=selector)

    assert np.array_equal(pop.global_best_position, pop.current_best_position)

@pytest.mark.parametrize("selector", selector_data)
def test_update_positions_by_population(pop_mod, selector):
    pop_mod.update_positions(level='population', selector=selector)

    print(pop_mod.fitness, pop_mod.previous_fitness, pop_mod.backup_fitness)

    if selector == 'all':
        assert float(pop_mod.fitness[0]) < float(pop_mod.previous_fitness[0])
        assert float(pop_mod.fitness[-1]) > float(pop_mod.previous_fitness[-1])
    elif selector == 'greedy':
        assert float(pop_mod.fitness[0]) < float(pop_mod.previous_fitness[0])
        assert float(pop_mod.fitness[-1]) == float(pop_mod.previous_fitness[-1])

@pytest.mark.parametrize("selector", selectors)
def test_update_positions_by_population_heterogeneous_selection(
        pop_mod, selector):
    sel = list(selector)
    pop_mod.update_positions(level='population', selector=sel)

    print(pop_mod.fitness, pop_mod.previous_fitness, pop_mod.backup_fitness)

    # [0] improves, [-1] worsens
    assert float(pop_mod.fitness[0]) < float(pop_mod.previous_fitness[0])

    print(sel)
    if sel[-1] == 'all':
        assert float(pop_mod.fitness[-1]) > float(pop_mod.previous_fitness[-1])
    elif sel[-1] == 'greedy':
        assert float(pop_mod.fitness[-1]) == float(pop_mod.previous_fitness[-1])

@pytest.mark.parametrize("selector", selector_global_data)
def test_update_positions_by_global(pop_mod, selector):
    pop_mod.update_positions(level='population', selector=selector)
    pop_mod.update_positions(level='global', selector=selector)

    if selector == 'greedy':
        assert float(pop_mod.global_best_fitness) < float(pop_mod.previous_global_best_fitness)
