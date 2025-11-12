""""
Test file
"""

import matplotlib.pyplot as plt
from sde_simulation import euler_maruyama, milstein

def main():
    # Simulation parameters
    x0 = 3  # Initial X value
    mu = 0.05   # Drift (5% per year)
    sigma = 0.2 # Volatility (20% per year)
    T = 1.0     # Time horizon (1 year)
    n_steps = 252 # Number of steps (e.g., trading days in a year)

    print("Running GBM simulation with C++ backend...")
    # Call the C++ function from Python
    path = milstein(x0, T=T, n_steps=n_steps, a=lambda x: 3-x, b=lambda x: .2)
    print("Simulation complete.")

    # Plot the results
    plt.plot(path)
    plt.title("SDE Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()