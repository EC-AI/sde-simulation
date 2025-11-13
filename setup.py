from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os

# A helper class to get pybind11 include paths
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Pybind11Extension(
        'sde_simulator_cpp',
        ['src/main.cpp', 'src/simulator.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            os.path.abspath('src'),
        ],
        language='c++'
    ),
]

setup(name='sde_simulator', ext_modules=ext_modules, version='0.0.3')