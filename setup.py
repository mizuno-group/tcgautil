# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:30:27 2020

@author: tadahaya
"""
from setuptools import setup,find_packages
from setuptools.command.test import test as TestCommand
import sys

__version__ = "0.0.0" # major.minor[.patch[.sub]] 

sys.path.append('.\\main')
sys.path.append('.\\tests')

REQUIRED_PKG = ['numpy','scipy','pandas','statsmodels','matplotlib','lifelines','sklearn'] # indicate required packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='PACKAGE_NAME', # add the package name
    version=__version__,
    description="a package for XXXX", # add short description
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mizuno-group',
    author='YOURNAME', # add your name
    author_email='tadahaya@gmail.com',
    license='MIT',
    classifiers=['Development Status :: 5 - Production/Stable'
                 ,'Environment :: Win32 (MS Windows)'
                 ,'Framework :: IPython'
                 ,'Intended Audience :: Science/Research'
                 ,'Operating System :: Microsoft :: Windows :: Windows 10'
                 ,'Programming Language :: Python :: 3'
                 ,'Topic :: Scientific/Engineering :: Bio-Informatics'],
    keywords=['omics','bioinformatics','transcriptome','chemoinformatics'],
    install_requires=REQUIRED_PKG,
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'enapy':['*.txt','*.ignore','*.ipynb','*.md']},
    packages=find_packages(exclude=('tests', 'docs')),
    zip_safe=False
    )