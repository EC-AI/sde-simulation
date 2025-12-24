""""
File for generating data for testing purposes.
"""

import numpy as np
import sde_simulation as sde
import matplotlib.pyplot as plt
import os
import argparse
import warnings

DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + "/data"
SEED = 42

# Default simulation parameters
x0 = 3
T = 10
n_steps = 252*T

time_steps = np.linspace(0, T, n_steps+1)

def a(x, t):
    return (3-x)*t**2

def b(x, t):
    return x**(.5)

def db_dx(x, t):
    return 0.5*x**(-.5)

def save_array(filename: str, array: np.ndarray):
    if os.path.exists(DATA_PATH + "/" + filename):
        warnings.warn("Trying to save a file that already exists. File will not be saved")
    else:
        np.savetxt(filename, array, delimiter=',')

def plot_data(em: np.ndarray, mil: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    ax.plot(time_steps, em, label="E-M")
    ax.plot(time_steps, mil, label="Milstein")
    ax.legend()
    ax.set_title("SDE Simulation")
    plt.show()

def generate_data(seed: int, plot: bool = False):
    gen = np.random.default_rng(seed)
    B_increments = gen.normal(0, np.sqrt(T/n_steps), n_steps)
    
    em = sde.euler_maruyama(x0, T, n_steps, a, b, B_increments)
    mil = sde.milstein(x0, T, n_steps, a, b, db_dx, B_increments)

    save_array('test_euler_maruyama', em)
    save_array('test_milstein', mil)

    if plot:
        plot_data(em, mil)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for testing purposes.")
    parser.add_argument('-p', '--plot', action='store_true', help="Plot the generated data.")
    args = parser.parse_args()
    generate_data(SEED, args.plot)