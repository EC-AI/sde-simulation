import pytest
import sde_simulator_cpp
import sde_simulation
import numpy as np

def test_module_import():
    assert sde_simulator_cpp is not None
    assert sde_simulator_cpp.__doc__ is not None
    assert sde_simulation is not None
    assert sde_simulator_cpp.euler_maruyama is not None
    assert sde_simulator_cpp.milstein is not None
    assert sde_simulation.euler_maruyama is not None
    assert sde_simulation.milstein is not None
    assert sde_simulation.euler_maruyama.__doc__ is not None
    assert sde_simulation.milstein.__doc__ is not None


class TestSimulation():
    
    # Default simulation parameters
    x0 = 3
    T = 10
    n_steps = 252*T

    @staticmethod
    def a(x, t):
        return (3-x)*t**2
    
    @staticmethod
    def b(x, t):
        return x**(.5)
    
    @staticmethod
    def db_dx(x, t):
        return 0.5*x**(-.5)

    def test_simulation(self):
        em = sde_simulation.euler_maruyama(self.x0, self.T, self.n_steps, self.a, self.b)
        mil = sde_simulation.milstein(self.x0, self.T, self.n_steps, self.a, self.b, self.db_dx)
        
        assert isinstance(em, np.ndarray)
        assert isinstance(mil, np.ndarray)

        assert em.shape == (self.n_steps+1,)
        assert mil.shape == (self.n_steps+1,)

    def test_seeded_simulation(self):
        SEED = 42
        EM_PATH = "tests/data/test_euler_maruyama"
        MIL_PATH = "tests/data/test_milstein"

        test_em = np.loadtxt(EM_PATH, delimiter=',')
        test_mil = np.loadtxt(MIL_PATH, delimiter=',')

        gen = np.random.default_rng(SEED)
        B_increments = gen.normal(0, np.sqrt(self.T/self.n_steps), self.n_steps)
        em = sde_simulation.euler_maruyama(self.x0, self.T, self.n_steps, self.a, self.b, B_increments)
        mil = sde_simulation.milstein(self.x0, self.T, self.n_steps, self.a, self.b, self.db_dx, B_increments)

        np.testing.assert_allclose(em, test_em)
        np.testing.assert_allclose(mil, test_mil)

    def test_malformed_function_calls(self):
        
        # Test invalid type for x0
        with pytest.raises(TypeError):
            sde_simulation.euler_maruyama("invalid", self.T, self.n_steps, self.a, self.b)
        with pytest.raises(TypeError):
            sde_simulation.milstein("invalid", self.T, self.n_steps, self.a, self.b, self.db_dx)

        # Test invalid type for T
        with pytest.raises(TypeError):
            sde_simulation.euler_maruyama(self.x0, "invalid", self.n_steps, self.a, self.b)
        with pytest.raises(TypeError):
            sde_simulation.milstein(self.x0, "invalid", self.n_steps, self.a, self.b, self.db_dx)

        # Test invalid type for n_steps
        with pytest.raises(TypeError):
            sde_simulation.euler_maruyama(self.x0, self.T, "invalid", self.a, self.b)
        with pytest.raises(TypeError):
            sde_simulation.milstein(self.x0, self.T, "invalid", self.a, self.b, self.db_dx)

        # Test invalid type for a
        with pytest.raises(TypeError):
            sde_simulation.euler_maruyama(self.x0, self.T, self.n_steps, "invalid", self.b)
        with pytest.raises(TypeError):
            sde_simulation.milstein(self.x0, self.T, self.n_steps, "invalid", self.b, self.db_dx)

        # Test invalid type for b
        with pytest.raises(TypeError):
            sde_simulation.euler_maruyama(self.x0, self.T, self.n_steps, self.a, "invalid")
        with pytest.raises(TypeError):
            sde_simulation.milstein(self.x0, self.T, self.n_steps, self.a, "invalid", self.db_dx)

        # Test invalid type for db_dx
        with pytest.raises(TypeError):
            sde_simulation.milstein(self.x0, self.T, self.n_steps, self.a, self.b, "invalid")

        # Test invalid type for B_increments
        with pytest.raises(TypeError):
            sde_simulation.euler_maruyama(self.x0, self.T, self.n_steps, self.a, self.b, "invalid")
        with pytest.raises(TypeError):
            sde_simulation.milstein(self.x0, self.T, self.n_steps, self.a, self.b, self.db_dx, "invalid")

        # Test invalid dimensions for Brownian increments
        wrong_increments = np.zeros(self.n_steps + 1)
        with pytest.raises(ValueError):
            sde_simulation.euler_maruyama(self.x0, self.T, self.n_steps, self.a, self.b, wrong_increments)
        with pytest.raises(ValueError):
            sde_simulation.milstein(self.x0, self.T, self.n_steps, self.a, self.b, self.db_dx, wrong_increments)
