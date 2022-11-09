from setuptools import setup, find_packages
from _version import __version__

def get_long_description():
    with open('README.md') as f:
        return f.read()

def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='berry',
    version=__version__,
    author='ADD AUTHOR',
    author_email='ADD EMAIL',
    url='ADD URL',
    description='ADD DESCRIPTION',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "networkx",
        "findiff",
        "matplotlib",
        "scipy",
        "argcomplete",
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: linux',
        'Topic :: Scientific/Engineering :: Physics',
    ],#TODO: Add classifiers
    entry_points={
        'console_scripts': [
            'berry = berry.cli:master_cli',
        ],
    },
)


