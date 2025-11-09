# Python SDE Simulator with C++ Backend

This project demonstrates how to accelerate Python code by implementing a computationally intensive task (SDE simulation) in C++ and exposing it to Python using `pybind11`.

## Prerequisites

You will need a C++ compiler and Python installed.

*   **On Windows**: Install Visual Studio with the "Desktop development with C++" workload.
*   **On macOS**: Install Xcode and its command-line tools.
*   **On Linux**: Install `g++` and `python-dev` (e.g., `sudo apt-get install build-essential python3-dev`).

## Setup and Installation

1.  **Navigate to the project directory:**

2.  **Install dependencies:**
    This project requires `pybind11`, `setuptools`, and `matplotlib`.
    ```sh
    pip install pybind11 setuptools matplotlib
    ```

3.  **Build the C++ extension:**
    Run the setup script. This will compile `main.cpp` into a Python module.
    ```sh
    python setup.py install
    ```

4.  **Run the simulation:**
    Execute the example Python script.
    ```sh
    python simulate.py
    ```

    This will run the simulation and display a plot of the resulting GBM path.