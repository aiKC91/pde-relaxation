import unit_test_manufactured as um

def test_smoke_convergence():
    err = um.run_smoke_test()
    # tolerance chosen to be generous for CI
    assert err < 1e-2
