import setuptools
import pathlib

# The text of the README file
README = (pathlib.Path(__file__).parent / 'README.md').read_text()

setuptools.setup(
    name='customhys',
    version='1.1',
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
    python_requires=">=3.8",
    include_package_data=True
)
