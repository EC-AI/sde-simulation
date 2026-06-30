""""
Test file
"""

import numpy as np
import sde_simulation as sde

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(f"Optional dependency Matplotlib is required for plotting. Please install it using 'pip install matplotlib' or install this package with optional ploting dependencies using 'pip install {sde.__name__}[plot]'.")

def main():
    # Simulation parameters
    x0 = 3     # Initial X value
    T = 10     # Time horizon
    n_steps = 10000 # Number of steps

    z = np.random.normal(0, (T/n_steps)**.5, n_steps)

    print("Running simulation with C++ backend...")
    
    em = sde.euler_maruyama(x0, T=T, n_steps=n_steps, a=lambda x, t: (3-x)*t**2, b=lambda x, t: x**(.5), B_increments=z)
    mil = sde.milstein(x0, T=T, n_steps=n_steps, a=lambda x, t: (3-x)*t**2, b=lambda x, t: x**(.5), db_dx=lambda x, t: 0.5*(x)**(-.5), B_increments=z)
    print("Simulation complete.")
    print(type(em))

    # Plot the results
    plt.plot(em, label="E-M")
    plt.plot(mil, label="Milstein", linewidth=0.5)
    plt.title("SDE Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()