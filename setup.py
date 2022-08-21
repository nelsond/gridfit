from setuptools import setup
import os
from gridfit.version import __version__

current_path = os.path.dirname(os.path.abspath(__file__))

# Get the long description from the README file
with open(os.path.join(current_path, 'README.md')) as f:
    long_description = f.read()

with open(os.path.join(current_path, 'requirements', 'common.txt')) as f:
    required = f.read().splitlines()

setup(
    name='gridfit',

    version=__version__,

    description='...',
    long_description=long_description,

    url='https://github.com/nelsond/gridfit',

    author='Nelson Darkwah Oppong',
    author_email='n@darkwahoppong.com',

    license='MIT',

    classifiers=[
        'Development Status :: 4 - Beta',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8'
    ],

    keywords='fit',

    packages=[
        'gridfit',
        'gridfit.drt',
        'gridfit.funcs',
        'gridfit.rect',
        'gridfit.roi',
        'gridfit.utils',
    ],

    install_requires=required
)
