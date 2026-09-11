import numpy as np
from unit_test_manufactured import run_smoke_test

def test_convergence_smoke():
    err = run_smoke_test()
    assert err < 1e-2
