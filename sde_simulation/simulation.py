
import sde_simulator_cpp as cpplib

def euler_maruyama(x0, T, n_steps, a=lambda x: 0, b=lambda x: 1):
    """
    Simulates a stochastic differential equation (SDE) using the Euler-Maruyama method.

    This method solves SDEs of the form:
    dX(t) = a(X(t), t)dt + b(X(t), t)dW(t)

    Args:
        x0 (float): The initial value of the process at t=0.
        T (float): The end time of the simulation.
        n_steps (int): The number of steps to use in the simulation.
        a (callable, optional): The drift function a(x). Defaults to a(x) = 0.
        b (callable, optional): The diffusion function b(x). Defaults to b(x) = 1.

    Returns:
        numpy.ndarray: A list of simulated values of the process at each time step.
    """
    return cpplib.euler_maruyama(x0, a, b, T, n_steps)

def milstein(x0, T, n_steps, a=lambda x: 0, b=None, db_dx=None):
    """
    Simulates a stochastic differential equation (SDE) using the Milstein scheme.

    This method is a higher-order method for solving SDEs of the form:
    dX(t) = a(X(t), t)dt + b(X(t), t)dW(t)

    It requires the derivative of the diffusion term with respect to x.

    Args:
        x0 (float): The initial value of the process at t=0.
        T (float): The end time of the simulation.
        n_steps (int): The number of steps to use in the simulation.
        a (callable, optional): The drift function a(x). Defaults to a(x) = 0.
        b (callable): The diffusion function b(x).
        db_dx (callable): The derivative of the diffusion function `b` with respect to x.

    Returns:
        numpy.ndarray: A list of simulated values of the process at each time step.

    Raises:
        ValueError: If b or db_dx are not provided.
    """
    if db_dx is None or b is None:
        raise ValueError("b and db_dx must be provided for Milstein scheme")
    return cpplib.milstein(x0, a, b, db_dx, T, n_steps)