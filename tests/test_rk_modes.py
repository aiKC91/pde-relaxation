import pytest
from spectral_solver import run_simulation
from scipy_solver import run_solver
import numpy as np


def test_rk_modes_small():
    psi_e = run_simulation(nx=32, ny=32, dt=0.005, t_end=0.01, rk='euler')
    psi_rk2 = run_simulation(nx=32, ny=32, dt=0.005, t_end=0.01, rk='rk2')
    psi_rk3 = run_simulation(nx=32, ny=32, dt=0.005, t_end=0.01, rk='rk3')
    assert psi_e.shape == psi_rk2.shape == psi_rk3.shape
    assert not np.isnan(psi_e).any()
    assert not np.isnan(psi_rk2).any()
    assert not np.isnan(psi_rk3).any()


def test_scipy_rk_option():
    p_e = run_solver(nx=32, ny=32, dt=0.005, t_end=0.01, rk='euler')
    p_rk3 = run_solver(nx=32, ny=32, dt=0.005, t_end=0.01, rk='rk3')
    assert p_e.shape == p_rk3.shape
    assert not np.isnan(p_rk3).any()
