from sde_simulator import simulate_gbm
import matplotlib.pyplot as plt

def main():
    # Simulation parameters
    s0 = 100.0  # Initial stock price
    mu = 0.05   # Drift (5% per year)
    sigma = 0.2 # Volatility (20% per year)
    T = 1.0     # Time horizon (1 year)
    n_steps = 252 # Number of steps (e.g., trading days in a year)

    print("Running GBM simulation with C++ backend...")
    # Call the C++ function from Python
    path = simulate_gbm(s0=s0, mu=mu, sigma=sigma, T=T, n_steps=n_steps)
    print("Simulation complete.")

    # Plot the results
    plt.plot(path)
    plt.title("Geometric Brownian Motion Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()