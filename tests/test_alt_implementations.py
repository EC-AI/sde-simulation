import os
import sys
import numpy as np
import pytest

# Ensure the tests directory is in sys.path so we can import alt_implementations
sys.path.insert(0, os.path.dirname(__file__))

from alt_implementations.raw_python_implementation import euler_maruyama as py_euler_maruyama
from sde_simulation import euler_maruyama as cpp_euler_maruyama

@pytest.mark.parametrize("x0, T, n_steps, a, b", [
    # Case 1: Constant drift and diffusion
    (1.0, 1.0, 100, lambda x, t: 0.1, lambda x, t: 0.2),
    # Case 2: Linear/GBM-like drift and diffusion
    (2.5, 3.0, 500, lambda x, t: 0.5 * x, lambda x, t: 0.3 * x),
    # Case 3: Non-linear time-dependent drift/diffusion
    (0.5, 2.0, 1000, lambda x, t: (2.0 - x) * t, lambda x, t: np.sqrt(np.abs(x)) * (1.0 + t)),
    # Case 4: Zero drift, constant diffusion
    (3.0, 5.0, 250, lambda x, t: 0, lambda x, t: 1),
])
def test_euler_maruyama_equivalence(x0, T, n_steps, a, b):
    # Use a fixed seed for reproducible random increments
    seed = 0
    rng = np.random.default_rng(seed)
    
    # Generate standard normal increments for the raw python version (z)
    z = rng.normal(0.0, 1.0, n_steps)
    
    # Calculate corresponding brownian increments (B_increments) for main module
    # main module expects B_increments as dW where Var(dW) = dt
    dt = T / n_steps
    B_increments = z * np.sqrt(dt)
    
    # Get results from raw python alternative implementation
    path_py = py_euler_maruyama(x0, z, T=T, a=a, b=b)
    
    # Get results from main C++-backed module
    path_cpp = cpp_euler_maruyama(
        x0=x0,
        T=T,
        n_steps=n_steps,
        a=a,
        b=b,
        B_increments=B_increments
    )
    
    # Verify that the trajectories are identical (within numerical precision)
    np.testing.assert_allclose(path_cpp, path_py, rtol=1e-12, atol=1e-12)


def test_euler_maruyama_defaults():
    # Test that default a and b lambda functions behave identically
    seed = 123
    rng = np.random.default_rng(seed)
    
    x0 = 1.5
    T = 1.0
    n_steps = 100
    
    z = rng.normal(0.0, 1.0, n_steps)
    dt = T / n_steps
    B_increments = z * np.sqrt(dt)
    
    # Calling both functions without specifying a and b to trigger their defaults
    path_py = py_euler_maruyama(x0, z, T=T)
    path_cpp = cpp_euler_maruyama(x0=x0, T=T, n_steps=n_steps, B_increments=B_increments)
    
    np.testing.assert_allclose(path_cpp, path_py, rtol=1e-12, atol=1e-12)
