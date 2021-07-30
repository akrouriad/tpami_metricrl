from setuptools import setup, find_packages
from codecs import open
from os import path

from metric_rl import __version__

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

long_description = 'Metric RL Meta is a Python Reinforcement Learning package' \
                   ' that implements the algorithms described in the paper:'\
                   ' \"Continuous Action Reinforcement Learning from a Mixture of Interpretable Experts\",' \
                   ' Riad Akrour, Davide Tateo, Jan Peters (2021).'

setup(
    name='metric-rl',
    version=__version__,
    description='A Python library for Interpretable RL.',
    long_description=long_description,
    author="Riad Akrour, Davide Tateo",
    author_email='davide@robot-learning.de',
    url='https://github.com/akrouriad/metricrl',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('metric_rl')],
    zip_safe=False,
    install_requires=requires_list,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ]
)
