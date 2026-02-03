# Contributing to customhys

Thank you for your interest in contributing to customhys! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/customhys.git
   cd customhys
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   make setup-dev
   # or manually:
   pip install -e ".[dev,ml,examples]"
   pre-commit install
   ```

## Development Workflow

### Code Quality

We use several tools to maintain code quality:

- **Black**: Code formatting (line length: 120)
- **Ruff**: Fast linting and code quality checks
- **MyPy**: Static type checking
- **Pytest**: Testing framework

Run all checks before committing:
```bash
make check-all
```

Or run individually:
```bash
make format      # Format code with black
make lint        # Check code with ruff
make typecheck   # Type check with mypy
make test        # Run tests
```

### Pre-commit Hooks

Pre-commit hooks will run automatically before each commit:
```bash
pre-commit install  # Install hooks
pre-commit run --all-files  # Run manually
```

### Testing

Write tests for new features and bug fixes:
```bash
make test        # Run tests with coverage
make test-fast   # Run tests without coverage
```

Tests should be placed in the `tests/` directory.

## Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run checks**
   ```bash
   make check-all
   make test
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Clear description of your changes"
   ```

5. **Push and create a Pull Request**
   ```bash
   git push origin your-branch-name
   ```

## Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Maximum line length: 120 characters
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Documentation
- Update README.md if adding new features
- Document all public APIs
- Include examples for complex functionality

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable

## Project Structure

```
customhys/
├── customhys/           # Main package
│   ├── __init__.py
│   ├── benchmark_func.py
│   ├── experiment.py
│   ├── hyperheuristic.py
│   ├── metaheuristic.py
│   ├── operators.py
│   ├── population.py
│   ├── tools.py
│   └── visualisation.py
├── tests/               # Test files
├── examples/            # Example notebooks and scripts
├── docfiles/            # Documentation assets
├── pyproject.toml       # Project configuration
├── setup.py            # Setup script
└── requirements.txt     # Core dependencies
```

## Optional Dependencies

The project has several optional dependency groups:

- `ml`: Machine Learning support (TensorFlow)
- `dev`: Development tools (pytest, black, ruff, mypy)
- `examples`: Jupyter notebooks support
- `docs`: Documentation building tools

Install with:
```bash
pip install -e ".[ml,dev,examples]"  # Install specific groups
pip install -e ".[all]"              # Install all optional dependencies
```

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Clear description of the problem
- Minimal code example to reproduce
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
