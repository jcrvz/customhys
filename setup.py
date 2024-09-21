# -*- coding: utf-8 -*-
import setuptools
import pathlib

# The text of the README file
README = (pathlib.Path(__file__).parent / 'README.md').read_text(encoding='utf-8')

setuptools.setup(
    name='customhys',
    version='1.1.7',
    packages=setuptools.find_packages(),
    url='https://github.com/jcrvz/customhys',
    license='MIT License',
    author='Jorge Mario Cruz-Duarte',
    author_email='jorge.cruz@tec.mx',
    description='This framework provides tools for solving, but not limited to, continuous optimisation problems '
                'using a hyper-heuristic approach for customising metaheuristics. ',
    long_description_content_type='text/markdown',
    long_description=README,
    keywords=['metaheuristics', 'hyper-heuristic', 'optimization', 'automatic design', 'global optimization',
              'evolutionary computation', 'bio-inspired', 'algorithm design'],
    python_requires=">=3.10",
    install_requires=(pathlib.Path(__file__).parent / 'requirements.txt').read_text(encoding='utf-8').split('\n'),
    extras_require={
        "ML": [
            "tensorflow>=2.8.0; sys_platform != 'darwin'",
            "tensorflow-macos>=2.10.0; sys_platform == 'darwin'",
            "tensorflow-metal>=0.7.1; sys_platform == 'darwin'",
        ]
    },
    include_package_data=True,

)
