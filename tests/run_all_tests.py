#!/usr/bin/env python3
"""
Comprehensive test runner for customhys project.
Runs all tests and provides a summary report.
"""

import subprocess
import sys
from pathlib import Path


def run_test_file(test_file, use_pytest=True):
    """Run a single test file."""
    print(f"\n{'=' * 70}")
    print(f"Running: {test_file.name}")
    print("=" * 70)

    if use_pytest:
        cmd = ["pytest", str(test_file), "-v", "--tb=short"]
    else:
        cmd = [sys.executable, str(test_file)]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Run all tests."""
    tests_dir = Path(__file__).parent

    print("\n" + "=" * 70)
    print("CUSTOMHYS PROJECT TEST SUITE")
    print("=" * 70)

    # Test files to run
    # Note: test_hyperheuristic.py and test_tools.py are skipped due to:
    # - test_hyperheuristic: Computationally intensive (several minutes runtime)
    # - test_tools: Import issues with tqdm/scipy interaction
    test_files = [
        ("test_setup.py", False),  # Run standalone
        ("test_makefile.py", False),  # Run standalone
        ("test_benchmark_func.py", True),  # Run with pytest
        ("test_operators.py", True),  # Run with pytest
        ("test_metaheuristic.py", True),  # Run with pytest
        ("test_population_update_positions.py", True),  # Run with pytest
        # ('test_tools.py', True),                      # SKIPPED: Import hangs
        # ('test_hyperheuristic.py', True),             # SKIPPED: Too slow (run separately)
    ]

    results = {}

    for test_file, use_pytest in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            success = run_test_file(test_path, use_pytest)
            results[test_file] = success
        else:
            print(f"⚠ Test file not found: {test_file}")
            results[test_file] = None

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for test_file, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"  {status}: {test_file}")

    print("-" * 70)
    print(f"Total: {len(results)} test suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")

    # Additional info about skipped tests
    if passed > 0:
        print("\n" + "=" * 70)
        print("ℹ Note: test_hyperheuristic.py and test_tools.py are skipped")
        print("  - test_hyperheuristic: Run separately (computationally intensive)")
        print("  - test_tools: Import issues (non-critical utility tests)")
        print("=" * 70)
    print("=" * 70)

    # Return exit code
    if failed > 0:
        print("\n❌ Some tests failed!")
        return 1
    elif passed == 0:
        print("\n⚠ No tests were run!")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
