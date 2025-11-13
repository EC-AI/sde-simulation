
import sde_simulator_cpp as cpplib

def euler_maruyama(x0, T, n_steps, a=lambda x: 0, b=lambda x: 1):
    return cpplib.euler_maruyama(x0, a, b, T, n_steps)

def milstein(x0, T, n_steps, a=lambda x: 0, b=lambda x: 1, db_dx=None):
    if db_dx is None:
        raise ValueError("db_dx must be provided for Milstein scheme")
    return cpplib.milstein(x0, a, b, db_dx, T, n_steps)