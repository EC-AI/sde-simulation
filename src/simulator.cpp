#include "simulator.hpp"
#include <random>
#include <cmath>

py::array_t<double> euler_maruyama(
    double x0, 
    const std::function<double(double, double)> &a, 
    const std::function<double(double, double)> &b, 
    double T, 
    int n_steps
) {
    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    double dt = T / n_steps;
    double sqrt_dt = std::sqrt(dt);
    double t;
    double dW;

    // Set up random number generation
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, sqrt(dt));

    for (int i = 0; i < n_steps; ++i) {
        t = i * dt;
        dW = distribution(generator)*sqrt_dt;
        path[i+1] = path[i] + a(path[i], t) * dt + b(path[i], t) * dW;
    }

    return py::array_t<double>(
        path.size(),
        path.data()
    );
}

py::array_t<double> milstein(
    double x0, 
    const std::function<double(double, double)> &a, 
    const std::function<double(double, double)> &b, 
    const std::function<double(double, double)> &db_dx, 
    double T, 
    int n_steps
) {
    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    double dt = T / n_steps;
    double sqrt_dt = std::sqrt(dt);
    double t;
    double dW;
    double bxt;

    // Set up random number generation
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, sqrt(dt));

    for (int i = 0; i < n_steps; ++i) {
        t = i * dt;
        dW = distribution(generator)*sqrt_dt;
        bxt = b(path[i], t);
        path[i+1] = path[i] + a(path[i], t) * dt + bxt * dW + 0.5 * bxt * db_dx(path[i], t) * (dW * dW - dt);
    }

    return py::array_t<double>(
        path.size(),
        path.data()
    );
}
