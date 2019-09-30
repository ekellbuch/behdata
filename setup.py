#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup



setup(
    name='behdata',
    version='0.1.0',
    description='Behavioral analysis in Video data',
    author='E. Kelly Buchanan',
    author_email='ekellbuchanan@gmail.com',
    url='https://github.com/ekellbuch/behdata',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/behdata/*.py', recursive=True)],
    include_package_data=True,
    zip_safe=False,
)