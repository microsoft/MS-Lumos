# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# !/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import find_packages
from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='mct',
    version='1.0.0',
    description="Tools to compare metrics between datasets, accounting for population differences "
                "and invariant features.",
    long_description=readme + '\n\n' + history,
    author="Jamie Pool, Ashkan Aazami, Ebrahim Beyrami, Jay Gupchup, Martin Ellis",
    author_email='',
    url='https://github.com/microsoft/MS-MCT',
    packages=find_packages(),
    package_dir={'mct': 'mct'},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords=['mct'],
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
