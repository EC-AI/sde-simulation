#include <vector>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * @brief Simulates a path of a Geometric Brownian Motion (GBM) process.
 * 
 * SDE: dS_t = mu * S_t * dt + sigma * S_t * dW_t
 * 
 * @param s0 Initial value of the process.
 * @param mu Drift rate.
 * @param sigma Volatility.
 * @param T Time horizon in years.
 * @param n_steps Number of time steps.
 * @return A vector of doubles representing the simulated path.
 */
std::vector<double> simulate_gbm(double s0, double mu, double sigma, double T, int n_steps) {
    std::vector<double> path(n_steps + 1);
    path[0] = s0;

    double dt = T / n_steps;

    // Set up random number generation
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, sqrt(dt));

    for (int i = 0; i < n_steps; ++i) {
        double dW = distribution(generator);
        path[i+1] = path[i] + mu * path[i] * dt + sigma * path[i] * dW;
    }

    return path;
}

PYBIND11_MODULE(sde_simulator, m) {
    m.doc() = "High-performance SDE simulator using C++"; // optional module docstring

    m.def("simulate_gbm", &simulate_gbm, "Simulates a Geometric Brownian Motion path.",
          py::arg("s0"),
          py::arg("mu"),
          py::arg("sigma"),
          py::arg("T"),
          py::arg("n_steps"));
}