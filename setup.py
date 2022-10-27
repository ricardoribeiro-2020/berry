from setuptools import setup, find_packages
from berry import __version__

def get_long_description():
    with open('README.md') as f:
        return f.read()

def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='berry',
    version=__version__,
    description='ADD DESCRIPTION',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='ADD AUTHOR',
    author_email='ADD EMAIL',
    url='ADD URL',
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],#TODO: Add classifiers
    entry_points={
        'console_scripts': [
            'berry = berry.cli:master_cli',
        ],
    },
)


