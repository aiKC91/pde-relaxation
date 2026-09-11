import numpy as np
from spectral_solver import run_simulation
from scipy_solver import run_solver


def test_rk_modes_small():
    for rk in ['euler', 'rk2', 'rk3']:
        psi = run_simulation(nx=16, ny=16, dt=0.01, t_end=0.02, rk=rk)
        assert psi.shape == (16, 16)
        assert np.isfinite(psi).all()


def test_scipy_rk_option():
    p_e = run_solver(nx=16, ny=16, dt=0.01, t_end=0.02, rk='euler')
    p_rk3 = run_solver(nx=16, ny=16, dt=0.01, t_end=0.02, rk='rk3')
    assert p_e.shape == p_rk3.shape
    assert not np.isnan(p_rk3).any()
