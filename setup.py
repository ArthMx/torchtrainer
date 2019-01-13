from setuptools import setup

setup(
    name='TorchTrainer',
    url='https://github.com/ArthMx/torchtrainer',
    author='Arthur Moraux',
    author_email='arthur.moraux@meteo.be',
    packages=['torchtrainer'],
    install_requires=['numpy', 'torch', 'matplotlib', 'collections', 'time', 
                      'sys', 'os', 'functools'],
    version='0.1',
    license='MIT',
    description='A package for training pytorch models.',
)