#include "simulator.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

py::array_t<double> euler_maruyama(
    double x0, 
    const std::function<double(double, double)> &a, 
    const std::function<double(double, double)> &b, 
    double T, 
    int n_steps,
    py::array_t<double> random_variates
) {
    py::buffer_info buf = random_variates.request();
    if (buf.size < n_steps) {
        throw std::runtime_error("Insufficient number of random variates provided.");
    }
    double *dW_ptr = static_cast<double *>(buf.ptr);

    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    double dt = T / n_steps;
    double t;

    for (int i = 0; i < n_steps; ++i) {
        t = i * dt;
        path[i+1] = path[i] + a(path[i], t) * dt + b(path[i], t) * dW_ptr[i];
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
    int n_steps,
    py::array_t<double> random_variates
) {
    py::buffer_info buf = random_variates.request();
    if (buf.size < n_steps) {
        throw std::runtime_error("Insufficient number of random variates provided.");
    }
    double *dW_ptr = static_cast<double *>(buf.ptr);

    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    double dt = T / n_steps;
    double t;
    double dW;
    double bxt;

    for (int i = 0; i < n_steps; ++i) {
        t = i * dt;
        dW = dW_ptr[i];
        bxt = b(path[i], t);
        path[i+1] = path[i] + a(path[i], t) * dt + bxt * dW + 0.5 * bxt * db_dx(path[i], t) * (dW * dW - dt);
    }

    return py::array_t<double>(
        path.size(),
        path.data()
    );
}
