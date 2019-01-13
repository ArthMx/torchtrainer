from setuptools import setup

setup(
    name='torchtrainer',
    version='0.1',
    description='A package for training pytorch models.',
    url='https://github.com/ArthMx/torchtrainer',
    author='Arthur Moraux',
    author_email='arthur.moraux@gmail.com',
    packages=['torchtrainer'],
    install_requires=['numpy', 'torch', 'matplotlib', 'functools'],
    license='MIT',
)
