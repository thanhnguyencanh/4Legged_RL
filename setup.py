#!/usr/bin/env python

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()
    print (len(required))
for i in range(len(required)):
    setup(name='rl_dev',
        version='1.0',
        description='Base dev environment for reinforcement learning projects',
        url='',
        packages=find_packages(),
        install_requires = required[i],
        long_description= ("Base dev environment for reinforcement learning projects")
        )
