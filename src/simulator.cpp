#include "simulator.hpp"
#include <random>
#include <cmath>

std::vector<double> euler_maruyama(
    double x0, 
    const std::function<double(double)> &a, 
    const std::function<double(double)> &b, 
    double T, 
    int n_steps
) {
    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    double dt = T / n_steps;

    // Set up random number generation
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, sqrt(dt));

    for (int i = 0; i < n_steps; ++i) {
        double dW = distribution(generator);
        path[i+1] = path[i] + a(path[i]) * dt + b(path[i]) * dW;
    }

    return path;
}

/**
 * @brief Simulates a path of an SDE of the form dX = a(X) dt + b(X) dW using Milstein's scheme.
 * 
 * @param x0 Initial value of the process.
 * @param a Drift function.
 * @param b Volatility function.
 * @param db_dx First derivative of b with respect to x.
 * @param T Time horizon.
 * @param n_steps Number of time steps.
 * @return A vector of doubles representing the simulated path.
 */
std::vector<double> milstein(
    double x0, 
    const std::function<double(double)> &a, 
    const std::function<double(double)> &b, 
    const std::function<double(double)> &db_dx, 
    double T, 
    int n_steps
) {
    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    double dt = T / n_steps;

    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, sqrt(dt));

    for (int i = 0; i < n_steps; ++i) {
        double dW = distribution(generator);
        path[i+1] = path[i] + a(path[i]) * dt + b(path[i]) * dW + 0.5 * b(path[i]) * db_dx(path[i]) * (dW * dW - dt);
    }

    return path;
}
