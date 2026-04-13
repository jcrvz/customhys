"""
Test suite to verify project setup and basic functionality.
This ensures dependencies, imports, and core features work correctly.
"""

import sys
from pathlib import Path


def test_python_version():
    """Verify Python version is 3.10 or higher."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version_info}"
    print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def test_core_imports():
    """Test that all core modules can be imported."""
    try:
        import matplotlib
        import numpy
        import pandas
        import scipy
        import sklearn
        import tqdm

        print(matplotlib)
        print(numpy)
        print(pandas)
        print(scipy)
        print(sklearn)
        print(tqdm)

        print("✓ All core dependencies imported successfully")
    except ImportError as e:
        raise AssertionError(f"Core dependency import failed: {e}") from e


def test_customhys_imports():
    """Test that all customhys modules can be imported."""
    try:
        from customhys import (
            benchmark_func,
            experiment,
            hyperheuristic,
            metaheuristic,
            operators,
            population,
            tools,
            visualisation,
        )

        print(benchmark_func)
        print(experiment)
        print(hyperheuristic)
        print(metaheuristic)
        print(operators)
        print(population)
        print(tools)
        print(visualisation)
        print("✓ All customhys modules imported successfully")
    except ImportError as e:
        raise AssertionError(f"Customhys module import failed: {e}") from e


def test_customhys_version():
    """Verify customhys version is correct."""
    import customhys

    assert hasattr(customhys, "__version__")
    assert customhys.__version__ == "1.1.9"
    print(f"✓ Customhys version: {customhys.__version__}")


def test_optional_ml_import():
    """Test TensorFlow import (optional dependency)."""
    try:
        import tensorflow as tf

        print(f"✓ TensorFlow available: {tf.__version__}")
    except ImportError:
        print("ℹ TensorFlow not installed (optional dependency)")


def test_benchmark_functions():
    """Test that benchmark functions are accessible."""
    from customhys import benchmark_func as bf

    # Test that __all__ is defined and has functions
    assert hasattr(bf, "__all__")
    assert len(bf.__all__) > 0

    # Test creating a simple benchmark function
    sphere = bf.Sphere(2)
    assert sphere is not None
    assert sphere.variable_num == 2
    print(f"✓ Benchmark functions available: {len(bf.__all__)} functions")


def test_population_creation():
    """Test creating a population."""
    from customhys import benchmark_func as bf
    from customhys import population as pp

    fun = bf.Sphere(2)
    pop = pp.Population(fun.get_search_range(), num_agents=10)

    assert pop.num_agents == 10
    assert pop.num_dimensions == 2
    print("✓ Population created successfully")


def test_operators_available():
    """Test that operators are available."""
    from customhys import operators as op

    assert hasattr(op, "__all__")
    assert len(op.__all__) > 0
    print(f"✓ Search operators available: {len(op.__all__)} operators")


def test_metaheuristic_creation():
    """Test creating a simple metaheuristic."""
    from customhys import metaheuristic as mh

    problem = {"function": lambda x: sum(x**2), "is_constrained": False, "boundaries": [[-100, 100], [-100, 100]]}

    # Simple search operator
    search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

    meta = mh.Metaheuristic(
        problem=problem, search_operators=search_operators, num_agents=10, num_iterations=5, verbose=False
    )

    assert meta is not None
    print("✓ Metaheuristic created successfully")


def test_simple_optimization():
    """Test a simple optimization run."""

    from customhys import benchmark_func as bf
    from customhys import metaheuristic as mh

    # Simple sphere function
    fun = bf.Sphere(2)

    search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

    meta = mh.Metaheuristic(
        problem=fun.get_formatted_problem(),
        search_operators=search_operators,
        num_agents=10,
        num_iterations=10,
        verbose=False,
    )

    # Run optimization
    meta.run()

    # Check that we have a result
    assert meta.historical is not None
    assert "fitness" in meta.historical
    assert len(meta.historical["fitness"]) > 0
    print(f"✓ Simple optimization completed (best fitness: {meta.historical['fitness'][-1]:.6f})")


def test_project_files_exist():
    """Verify important project files exist."""
    project_root = Path(__file__).parent.parent

    required_files = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "Makefile",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
    ]

    for file_name in required_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"Required file missing: {file_name}"

    print(f"✓ All {len(required_files)} required project files exist")


def test_uv_lock_exists():
    """Verify uv.lock file exists (for UV project management)."""
    project_root = Path(__file__).parent.parent
    uv_lock = project_root / "uv.lock"

    if uv_lock.exists():
        print(f"✓ uv.lock exists ({uv_lock.stat().st_size / 1024:.1f} KB)")
    else:
        print("ℹ uv.lock not found (not using UV project management)")


def test_package_data_files():
    """Verify package data files exist."""
    from pathlib import Path

    import customhys

    customhys_path = Path(customhys.__file__).parent
    collections_path = customhys_path / "collections"

    assert collections_path.exists(), "collections directory not found"

    # Check for some collection files
    collection_files = list(collections_path.glob("*.txt"))
    assert len(collection_files) > 0, "No collection files found"
    print(f"✓ Package data files present ({len(collection_files)} collection files)")


if __name__ == "__main__":
    """Run all tests when executed directly."""

    print("=" * 60)
    print("Running Project Setup Tests")
    print("=" * 60)

    # Get all test functions
    test_functions = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]

    passed = 0
    failed = 0

    for test_func in test_functions:
        print(f"\n{test_func.__name__}:")
        print("-" * 60)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
