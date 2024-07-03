#!/usr/bin/env python

import os
from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements_filename():
    if 'READTHEDOCS' in os.environ:
        return "REQUIREMENTS-RTD.txt"
    elif 'DOCKER' in os.environ:
        return "REQUIREMENTS-DOCKER.txt"
    else:
        return "REQUIREMENTS.txt"


install_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), get_requirements_filename()))
]

rtd_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), "REQUIREMENTS-RTD.txt"))
]

setup(
    name='cellmincer',
    version='0.1.0',
    description='A software package for learning self-supervised denoising models for voltage-imaging movies',
    long_description=readme(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research'
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 3.10',
      'Topic :: Scientific/Engineering',
    ],
    url='http://github.com/broadinstitute/CellMincer',
    author='Brice Wang, Mehrtash Babadi',
    license='BSD (3-Clause)',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        'dev': rtd_requires + [
            'docs',
            'lint',
            'mypy',
            'ruff',
        ],
    },
    entry_points={
        'console_scripts': ['cellmincer=cellmincer.cli.base_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)