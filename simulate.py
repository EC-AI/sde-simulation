import sde_simulator_cpp as cpplib
import matplotlib.pyplot as plt

def euler_maruyama(x0, T, n_steps, a=lambda x: 0, b=lambda x: 1):
    return cpplib.euler_maruyama(x0, a, b, T, n_steps)



def main():
    # Simulation parameters
    x0 = 2  # Initial X value
    mu = 0.05   # Drift (5% per year)
    sigma = 0.2 # Volatility (20% per year)
    T = 1.0     # Time horizon (1 year)
    n_steps = 252 # Number of steps (e.g., trading days in a year)

    print("Running GBM simulation with C++ backend...")
    # Call the C++ function from Python
    path = euler_maruyama(x0, T=T, n_steps=n_steps, a=lambda x: 3-x, b=lambda x: 3 * x**.5)
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