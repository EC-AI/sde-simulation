from setuptools import setup
import pybind11
from pybind11.setup_helpers import Pybind11Extension
import os

ext_modules = [
    Pybind11Extension(
        'sde_simulator_cpp',
        ['src/main.cpp', 'src/simulator.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(True),
            os.path.abspath('src'),
        ],
        language='c++'
    ),
]

setup(name='sde_simulator', ext_modules=ext_modules)