
import sde_simulator_cpp as cpplib
from typing import Callable, Optional
import numpy as np
from numpy.typing import ArrayLike

def euler_maruyama(
        x0: float,
        T: float,
        n_steps: int,
        a: Optional[Callable[[float, float], float]] = lambda x, t: 0,
        b: Optional[Callable[[float, float], float]] = lambda x, t: 1,
        B_increments: Optional[ArrayLike] = None
    ) -> np.ndarray:
    """
    Simulates a stochastic differential equation (SDE) using the Euler-Maruyama method.

    This method solves SDEs of the form:
    dX(t) = a(X(t), t)dt + b(X(t), t)dW(t)

    Args:
        x0 (float): The initial value of the process at t=0.
        T (float): The end time of the simulation.
        n_steps (int): The number of steps to use in the simulation.
        a (callable, optional): The drift function a(x, t). Defaults to a(x, t) = 0.
        b (callable, optional): The diffusion function b(x, t). Defaults to b(x, t) = 1.
        B_increments (ArrayLike, optional): Increments for the driving process to use for integration. If `None` it will generate random variates with mean 0 and variance `T/n_steps`.

    Returns:
        numpy.ndarray: A list of simulated values of the process at each time step.
    """
    if B_increments is not None:
        B_increments = np.asarray(B_increments)
    else:
        dt = T / n_steps
        B_increments = np.random.normal(0.0, np.sqrt(dt), n_steps)
    return cpplib.euler_maruyama(x0, a, b, T, n_steps, B_increments)

def milstein(
        x0: float,
        T: float,
        n_steps: int,
        a: Optional[Callable[[float, float], float]] = lambda x, t: 0,
        b: Callable[[float, float], float] = None,
        db_dx: Callable[[float, float], float] = None,
        B_increments: Optional[ArrayLike] = None,
    ) -> np.ndarray:
    """
    Simulates a stochastic differential equation (SDE) using the Milstein scheme.

    This method is a higher-order method for solving SDEs of the form:
    dX(t) = a(X(t), t)dt + b(X(t), t)dW(t)

    It requires the derivative of the diffusion term with respect to x.

    Args:
        x0 (float): The initial value of the process at t=0.
        T (float): The end time of the simulation.
        n_steps (int): The number of steps to use in the simulation.
        a (callable, optional): The drift function a(x, t). Defaults to a(x, t) = 0.
        b (callable): The diffusion function b(x, t).
        db_dx (callable): The derivative of the diffusion function `b` with respect to x, db/dx(x, t).
        B_increments (ArrayLike, optional): Increments for the driving process to use for integration. If `None` it will generate random variates with mean 0 and variance `T/n_steps`.

    Returns:
        numpy.ndarray: A list of simulated values of the process at each time step.

    Raises:
        ValueError: If b or db_dx are not provided.
    """
    if db_dx is None or b is None:
        raise ValueError("b and db_dx must be provided for Milstein scheme")
    if B_increments is not None:
        B_increments = np.asarray(B_increments)
    else:
        dt = T / n_steps
        B_increments = np.random.normal(0.0, np.sqrt(dt), n_steps)
    return cpplib.milstein(x0, a, b, db_dx, T, n_steps, B_increments)