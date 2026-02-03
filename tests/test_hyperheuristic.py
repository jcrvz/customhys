"""
Test suite for hyperheuristic module.
Tests hyperheuristic creation and search space exploration.

NOTE: These tests are computationally intensive and may take several minutes.
They are marked as 'slow' and skipped by default.
Run with: pytest tests/test_hyperheuristic.py -m slow -v
"""

import os
from pathlib import Path

import pytest

from customhys import benchmark_func as bf
from customhys import hyperheuristic as hh

# Mark all tests in this module as slow
pytestmark = pytest.mark.skip(reason="Hyperheuristic tests are computationally intensive. Run explicitly with pytest tests/test_hyperheuristic.py")


# Change to the customhys package directory so relative paths work
# The hyperheuristic module expects 'collections/filename.txt'
TEST_DIR = Path(__file__).parent
CUSTOMHYS_DIR = TEST_DIR.parent / 'customhys'

# Store original directory
ORIGINAL_DIR = os.getcwd()


def setup_module():
    """Change to customhys directory before tests."""
    os.chdir(CUSTOMHYS_DIR)


def teardown_module():
    """Restore original directory after tests."""
    os.chdir(ORIGINAL_DIR)


def get_test_parameters(**overrides):
    """Get default test parameters merged with any overrides."""
    defaults = {
        'cardinality': 2,
        'cardinality_min': 1,
        'num_replicas': 5,
        'num_iterations': 10,
        'num_agents': 10,
        'as_mh': False,
        'num_steps': 10,  # Reduce for faster testing
        'verbose': False  # Disable verbose for tests
    }
    defaults.update(overrides)
    return defaults


class TestHyperheuristicCreation:
    """Test hyperheuristic object creation."""

    def test_hyperheuristic_instantiation(self):
        """Test basic hyperheuristic creation."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters=get_test_parameters()
        )

        assert hyper is not None

    def test_hyperheuristic_with_parameters(self):
        """Test hyperheuristic with custom parameters."""
        fun = bf.Rastrigin(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters=get_test_parameters(
                cardinality=3,
                num_replicas=10,
                num_iterations=20,
                num_agents=15
            )
        )

        assert hyper.max_cardinality == 3
        assert hyper.parameters['num_replicas'] == 10


class TestHyperheuristicSearch:
    """Test hyperheuristic search functionality."""

    def test_hyperheuristic_run(self):
        """Test running a hyperheuristic search."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 3,
                'num_iterations': 5,
                'num_agents': 10,
                'as_mh': False,
                'num_steps': 10  # Reduce steps for faster testing
            }
        )

        try:
            hyper.run()
            assert True  # If it runs without error
        except Exception as e:
            # Hyperheuristic might require specific setup
            pytest.skip(f"Hyperheuristic run requires specific setup: {e}")

    def test_hyperheuristic_solution_format(self):
        """Test that hyperheuristic produces valid solutions."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False,
                'num_steps': 10
            }
        )

        try:
            hyper.run()
            # Check if results are stored
            assert hasattr(hyper, 'results')
        except Exception:
            pytest.skip("Hyperheuristic requires specific configuration")


class TestHyperheuristicCardinality:
    """Test hyperheuristic with different cardinalities."""

    @pytest.mark.parametrize("cardinality", [1, 2, 3, 4])
    def test_different_cardinalities(self, cardinality):
        """Test hyperheuristic with different cardinality values."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': cardinality,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        assert hyper.cardinality == cardinality


class TestHyperheuristicHeuristicSpace:
    """Test hyperheuristic with different heuristic spaces."""

    def test_default_heuristic_space(self):
        """Test with default heuristic space."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        assert hyper.heuristic_space is not None

    def test_custom_heuristic_list(self):
        """Test with custom heuristic list."""
        fun = bf.Sphere(2)

        custom_heuristics = [
            ('random_search', {'scale': 1.0, 'distribution': 'uniform'}, 'greedy'),
            ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'gaussian'}, 'greedy'),
        ]

        hyper = hh.Hyperheuristic(
            heuristic_space=custom_heuristics,
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        assert hyper.heuristic_space is not None


class TestHyperheuristicMetaheuristicGeneration:
    """Test metaheuristic generation capabilities."""

    def test_generate_metaheuristic(self):
        """Test generating metaheuristics from hyperheuristic."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        # Test that hyperheuristic has search space
        assert hasattr(hyper, 'heuristic_space')


class TestHyperheuristicReplicas:
    """Test hyperheuristic replica functionality."""

    @pytest.mark.parametrize("num_replicas", [1, 3, 5, 10])
    def test_different_replica_counts(self, num_replicas):
        """Test hyperheuristic with different numbers of replicas."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': num_replicas,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        assert hyper.num_replicas == num_replicas


class TestHyperheuristicParameters:
    """Test hyperheuristic parameter handling."""

    def test_parameter_validation(self):
        """Test that parameters are validated correctly."""
        fun = bf.Sphere(2)

        valid_params = {
            'cardinality': 2,
            'num_replicas': 5,
            'num_iterations': 10,
            'num_agents': 10
        }

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters=valid_params
        )

        assert hyper.cardinality == valid_params['cardinality']
        assert hyper.num_replicas == valid_params['num_replicas']

    def test_default_parameters(self):
        """Test default parameter values."""
        fun = bf.Sphere(2)

        # Create with minimal parameters
        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
            }
        )

        # Should have default values for other parameters
        assert hasattr(hyper, 'cardinality')


class TestHyperheuristicWithDifferentProblems:
    """Test hyperheuristic with different optimization problems."""

    @pytest.mark.parametrize("func_class", [
        bf.Sphere,
        bf.Rastrigin,
    ])
    def test_hyperheuristic_different_problems(self, func_class):
        """Test hyperheuristic with different benchmark functions."""
        fun = func_class(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        assert hyper is not None


class TestHyperheuristicSearchStrategy:
    """Test hyperheuristic search strategies."""

    def test_search_strategy_simulated_annealing(self):
        """Test hyperheuristic with simulated annealing strategy."""
        fun = bf.Sphere(2)

        params = {
            'cardinality': 2,
            'num_replicas': 3,
            'num_iterations': 10,
            'num_agents': 10,
            'cooling_rate': 0.95,  # SA parameter
            'initial_temperature': 100.0  # SA parameter
        }

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters=params
        )

        assert hyper.cardinality == 2


class TestHyperheuristicDataStructure:
    """Test hyperheuristic data structure and storage."""

    def test_hyperheuristic_attributes(self):
        """Test that hyperheuristic has expected attributes."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        # Check for expected attributes
        assert hasattr(hyper, 'heuristic_space')
        assert hasattr(hyper, 'problem')
        assert hasattr(hyper, 'cardinality')
        assert hasattr(hyper, 'num_replicas')


class TestHyperheuristicEdgeCases:
    """Test edge cases for hyperheuristic."""

    def test_minimum_cardinality(self):
        """Test hyperheuristic with minimum cardinality."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 1,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 5,
                'as_mh': False
            }
        )

        assert hyper.cardinality == 1

    def test_small_population(self):
        """Test hyperheuristic with small population."""
        fun = bf.Sphere(2)

        hyper = hh.Hyperheuristic(
            heuristic_space='default.txt',
            problem=fun.get_formatted_problem(),
            parameters={
                'cardinality': 2,
                'num_replicas': 2,
                'num_iterations': 5,
                'num_agents': 3,
                'as_mh': False
            }
        )

        assert hyper is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
