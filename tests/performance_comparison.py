from timeit import Timer
import numpy as np
from sde_simulation import euler_maruyama, milstein

# PARAMETERS
x0 = 3            # Initial X value
T = 10            # Time horizon
n_steps = 252*T   # Number of steps

a=lambda x, t: (3-x)*t**2         # Drift function
b=lambda x, t: x**(.5)            # Diffusion function
db_dx=lambda x, t: .5*(x)**(-.5)  # Derivative of b w.r.t. x

# Global variables to pass to timeit
globals = {
    'x0': x0,
    'T': T,
    'n_steps': n_steps,
    'a': a,
    'b': b,
    'db_dx': db_dx,
    'euler_maruyama': euler_maruyama,
    'milstein': milstein,
}

t_em = Timer(
    stmt='euler_maruyama(x0, T=T, n_steps=n_steps, a=a, b=b)',
    globals=globals,
)
t_mil = Timer(
    stmt='milstein(x0, T=T, n_steps=n_steps, a=a, b=b, db_dx=db_dx)',
    globals=globals,
)

number = 1000 # Number of times timeit will run the statement

em_times = [total/number for total in t_em.repeat(3, number=number)]
mil_times = [total/number for total in t_mil.repeat(3, number=number)]

print(f"\n{'Method':<20} | {'Mean (s)':<15} | {'Min (s)':<15}")
print("-" * 50)

for name, times in [("Euler-Maruyama", em_times), ("Milstein", mil_times)]:
    mean_t = np.mean(times)
    min_t = min(times)
    print(f"{name:<20} | {mean_t:.6f}        | {min_t:.6f}")
