#include <vector>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <functional>

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


/**
 * @brief Simulates a path of an SDE of the form dX = a dt + b dB using Euler's method.
 * 
 * @param x0 Initial value of the process.
 * @param a Drift function.
 * @param b Volatility function.
 * @param T Time horizon.
 * @param n_steps Number of time steps.
 * @return A vector of doubles representing the simulated path.
 */
std::vector<double> euler_maruyama(double x0, std::function<double(double)> &a, std::function<double(double)> &b, double T, int n_steps) {
    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    double dt = T / n_steps;

    // Set up random number generation
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, sqrt(dt));

    for (int i = 0; i < n_steps; ++i) {
        double dW = distribution(generator);
        path[i+1] = path[i] + a(path[i]) * path[i] * dt + b(path[i]) * path[i] * dW;
    }

    return path;
}


PYBIND11_MODULE(sde_simulator_cpp, m) {
    m.doc() = "High-performance SDE simulator using C++"; // optional module docstring

    m.def("simulate_gbm", &simulate_gbm, "Simulates a Geometric Brownian Motion path.",
          py::arg("s0"),
          py::arg("mu"),
          py::arg("sigma"),
          py::arg("T"),
          py::arg("n_steps"));

    m.def("euler_maruyama", &euler_maruyama, 
        "Simulates a path of an SDE of the form dX = a dt + b dB using Euler's method.",
          py::arg("x0"),
          py::arg("a"),
          py::arg("b"),
          py::arg("T"),
          py::arg("n_steps"));
}