# -*- coding: utf-8 -*-
import setuptools
import pathlib

# The text of the README file
README = (pathlib.Path(__file__).parent / 'README.md').read_text(encoding='utf-8')

setuptools.setup(
    name='customhys',
    version='1.1.8',
    packages=setuptools.find_packages(),
    url='https://github.com/jcrvz/customhys',
    license='MIT License',
    author='Jorge Mario Cruz-Duarte',
    author_email='jorge.cruz-duarte@univ-lille.fr',
    description='This framework provides tools for solving, but not limited to, continuous optimisation problems '
                'using a hyper-heuristic approach for customising metaheuristics.',
    long_description_content_type='text/markdown',
    long_description=README,
    keywords=['metaheuristics', 'hyper-heuristic', 'optimization', 'automatic design', 'global optimization',
              'evolutionary computation', 'bio-inspired', 'algorithm design'],
    python_requires=">=3.10",
    install_requires=[
        line.strip() for line in
        (pathlib.Path(__file__).parent / 'requirements.txt').read_text(encoding='utf-8').split('\n')
        if line.strip() and not line.strip().startswith('#')
    ],
    extras_require={
        "ml": [
            "tensorflow>=2.16.0; sys_platform != 'darwin'",
            "tensorflow-macos>=2.16.0; sys_platform == 'darwin'",
            "tensorflow-metal>=1.1.0; sys_platform == 'darwin'",
        ],
        "dev": [
            "pytest>=8.3.0",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "ruff>=0.6.0",
            "mypy>=1.11.0",
            "pre-commit>=3.8.0",
        ],
        "examples": [
            "jupyter>=1.1.0",
            "jupyterlab>=4.2.0",
            "ipywidgets>=8.1.0",
            "notebook>=7.2.0",
        ],
    },
    include_package_data=True,

)
