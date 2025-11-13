#include "simulator.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
 
namespace py = pybind11;


PYBIND11_MODULE(sde_simulator_cpp, m) {
    m.doc() = "High-performance SDE simulator using C++";

    m.def("euler_maruyama", &euler_maruyama, 
        "Simulates a path of an SDE of the form dX = a(X) dt + b(X) dW using Euler's method.",
        py::arg("x0"),
        py::arg("a"),
        py::arg("b"),
        py::arg("T"),
        py::arg("n_steps"));

    m.def("milstein", &milstein, 
        "Simulates a path of an SDE of the form dX = a(X) dt + b(X) dW using Milstein's scheme.",
        py::arg("x0"),
        py::arg("a"),
        py::arg("b"),
        py::arg("db_dx"),
        py::arg("T"),
        py::arg("n_steps"));
}

