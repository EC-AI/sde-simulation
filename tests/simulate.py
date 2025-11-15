""""
Test file
"""

import sde_simulation as sde

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(f"Optional dependency Matplotlib is required for plotting. Please install it using 'pip install matplotlib' or install this package with optional ploting dependencies using 'pip install {sde.__name__}[plot]'.")

def main():
    # Simulation parameters
    x0 = 3  # Initial X value
    mu = 0.05   # Drift (5% per year)
    sigma = 0.2 # Volatility (20% per year)
    T = 10     # Time horizon (1 year)
    n_steps = 252*T # Number of steps (e.g., trading days in a year)

    print("Running simulation with C++ backend...")
    
    em = sde.euler_maruyama(x0, T=T, n_steps=n_steps, a=lambda x, t: (3-x)*t**2, b=lambda x, t: x**(1/2))
    mil = sde.milstein(x0, T=T, n_steps=n_steps, a=lambda x, t: (3-x)*t**2, b=lambda x, t: x**(1/2), db_dx=lambda x, t: 0.5*(x)**(-.5))
    print("Simulation complete.")
    print(type(em))

    # Plot the results
    plt.plot(em, label="E-M")
    plt.plot(mil, label="Milstein")
    plt.title("SDE Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()