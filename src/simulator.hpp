#pragma once

#include <vector>
#include <functional>

/**
 * @brief Simulates a path of an SDE of the form dX = a(X) dt + b(X) dW using Euler's method.
 * 
 * @param x0 Initial value of the process.
 * @param a Drift function.
 * @param b Volatility function.
 * @param T Time horizon.
 * @param n_steps Number of time steps.
 * @return A vector of doubles representing the simulated path.
 */
std::vector<double> euler_maruyama(
    double x0, 
    const std::function<double(double)> &a, 
    const std::function<double(double)> &b, 
    double T, 
    int n_steps
);

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
);