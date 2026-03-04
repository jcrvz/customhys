#!/usr/bin/env python3
"""
Quick validation script to verify project setup.
Run this to check if everything is working correctly.
"""

import subprocess
import sys
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def check_item(description, condition, details=None):
    """Check an item and print result."""
    status = "✓" if condition else "✗"
    print(f"{status} {description}")
    if details:
        print(f"  → {details}")
    return condition


def run_command(cmd):
    """Run a command and return success status."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def main():
    """Run validation checks."""
    print_header("CUSTOMHYS PROJECT VALIDATION")

    all_passed = True

    # Check Python version
    print_header("Python Environment")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    all_passed &= check_item("Python 3.10+", sys.version_info >= (3, 10), f"Version: {python_version}")

    # Check UV installation
    print_header("Build Tools")
    uv_installed, uv_version, _ = run_command("uv --version")
    all_passed &= check_item(
        "UV installed (optional but recommended)",
        uv_installed,
        uv_version if uv_installed else "Not installed - using pip",
    )

    make_installed, _, _ = run_command("command -v make")
    all_passed &= check_item(
        "Make available", make_installed, "Makefile commands available" if make_installed else "Install make"
    )

    # Check project files
    print_header("Project Files")
    # Get the actual project root (parent of this script)
    project_root = Path(__file__).parent

    critical_files = {
        "pyproject.toml": "Project configuration",
        "setup.py": "Setup script",
        "requirements.txt": "Dependencies",
        "Makefile": "Build commands",
        "README.md": "Documentation",
    }

    for filename, description in critical_files.items():
        exists = (project_root / filename).exists()
        all_passed &= check_item(f"{filename}", exists, description)

    # Check optional files
    optional_files = {
        "uv.lock": "UV lockfile (for reproducibility)",
        "CHANGELOG.md": "Version history",
        "CONTRIBUTING.md": "Contribution guidelines",
    }

    print()
    for filename, description in optional_files.items():
        exists = (project_root / filename).exists()
        check_item(f"{filename} (optional)", exists, description)

    # Check imports
    print_header("Package Imports")

    try:
        import customhys

        check_item("customhys package", True, f"Version {customhys.__version__}")

        # Test core modules
        modules = ["benchmark_func", "population", "operators", "metaheuristic"]
        for module in modules:
            try:
                exec(f"from customhys import {module}")
                check_item(f"customhys.{module}", True)
            except ImportError:
                check_item(f"customhys.{module}", False)
                all_passed = False
    except ImportError as e:
        check_item("customhys package", False, str(e))
        all_passed = False

    # Check dependencies (with error handling)
    print_header("Core Dependencies")
    core_deps = ["numpy", "scipy", "matplotlib", "pandas", "sklearn"]

    for dep in core_deps:
        try:
            __import__(dep)
            check_item(dep, True)
        except (ImportError, ValueError):
            # ValueError can occur with numpy/pandas version mismatch
            check_item(dep, False, "Import error (may need reinstall)")
            if dep in ["numpy", "pandas"]:
                print(f"  ℹ Hint: Try 'uv sync' or 'pip install --upgrade {dep}'")

    # Check optional dependencies
    print()
    optional_deps = {
        "tensorflow": "Machine Learning",
        "pytest": "Testing",
        "black": "Code Formatting",
        "ruff": "Linting",
    }

    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            check_item(f"{dep} (optional)", True, description)
        except ImportError:
            check_item(f"{dep} (optional)", False, f"{description} - not installed")

    # Test Makefile
    if make_installed:
        print_header("Makefile Commands")

        success, output, _ = run_command("make help")
        check_item("make help", success, "Help command works")

        success, output, _ = run_command("make check-uv")
        check_item("make check-uv", success, output)

    # Summary
    print_header("VALIDATION SUMMARY")

    if all_passed:
        print("✅ All critical checks passed!")
        print("\nYou can now:")
        print("  • Run tests: make test")
        print("  • Format code: make format")
        print("  • Check code: make lint")
        print("  • See all commands: make help")
        return 0
    else:
        print("⚠ Some checks failed!")
        print("\nTo fix:")
        if not uv_installed:
            print("  • Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  • Install dependencies: make sync (or make install-dev)")
        print("  • Check documentation: README.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
