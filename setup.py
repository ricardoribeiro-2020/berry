from setuptools import find_packages, setup

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
    author='Berry Developers',
    author_email='ricardo.ribeiro@physics.org',
    url='https://ricardoribeiro-2020.github.io/berry/',
    description='The berry suite of programs extracts the Bloch wavefunctions from DFT calculations in an ordered way so they can be directly used to make calculations.',
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
        'Programming Language :: Python :: 3 :: Only',
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


