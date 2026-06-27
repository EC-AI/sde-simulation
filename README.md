# Python SDE Simulator with C++ Backend

This project demonstrates how to accelerate Python code by implementing a computationally intensive task (SDE simulation) in C++ and exposing it to Python using `pybind11`.

## Setup and Installation

1. Download the repo
2. Navigate to the project directory
3. Run 
```sh
pip install .
```
4. You can run the example script which simulates and plots a process
```sh
python ./tests/simulate.py
```

## Usage

You can use the simulator in python by importing it. The two main functions are `euler_maruyama` and `milstein`, which solve the SDE $dX_t = a(X_t, t) dt + b(X_t, t) dB_t$ using the Euler-Maruyama and Milstein schemes, respectively.

```python
import sde_simulation

em = sde_simulation.euler_maruyama(x0, T, n_steps, a, b, B_increments)
mil = sde_simulation.milstein(x0, T, n_steps, a, b, db_dx, B_increments)
```
Where `x0` is the initial value, `T` is the time horizon, `n_steps` is the number of steps, `a` is the drift function, `b` is the diffusion function, `db_dx` is the derivative of the diffusion function with respect to `x`, and `B_increments` (optional) is the sequence of Brownian increments. If `B_increments` is not provided, they will be generated using `numpy.random` with a standard normal distribution and variance equal to `T/n_steps`.

The function returns a numpy array of length `n_steps + 1` containing the values of the SDE at each time step.

Example:

```python
import sde_simulation
import numpy as np

# Define the SDE parameters
x0 = 1.0
T = 1.0
n_steps = 1000

a = lambda x, t: 0.1 * x  # Drift term
b = lambda x, t: 0.2    # Diffusion term
db_dx = lambda x, t: 0.0  # Derivative of diffusion term

# Generate Brownian increments
rng = np.random.default_rng(0)
B_increments = rng.normal(0, np.sqrt(T/n_steps), n_steps)

# Run the simulations
em_path = sde_simulation.euler_maruyama(x0, T, n_steps, a, b, B_increments)
mil_path = sde_simulation.milstein(x0, T, n_steps, a, b, db_dx, B_increments)

# Print the results
print("Euler-Maruyama path:", em_path)
print("Milstein path:", mil_path)
```
