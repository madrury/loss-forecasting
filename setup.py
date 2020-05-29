from setuptools import setup, find_packages
from os import path

setup(
    name='loss-development',
    version='0.0.1',
    description='Model Forecasting for Loss Development Curves',
    author='Matthew Drury',
    author_email='mdrury@remitly.com',
    license=None,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
    packages=['loss_development'],
    install_requires=[
        'numpy',
        'pandas'
    ],
)