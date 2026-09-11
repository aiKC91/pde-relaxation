# PDE relaxation repository

This repository contains example solvers and tests for the PDE-based relaxation equation:

    dPsi/dt + u · ∇Psi = alpha ∇² Psi − beta Psi ||∇×u|| + gamma F

The goal is to replace discrete grid-reallocation/remeshing steps with a continuous PDE operator that is local and parallelizable.

Files included:
- scipy_solver.py — SciPy-based semi-Lagrangian + implicit diffusion solver (cubic interp, RK2 backtrace)
- spectral_solver.py — FFT spectral implicit diffusion solver (fast for periodic domains)
- fipy_solver.py — FiPy finite-volume example (operator-split, optional)
- gpu_spectral_example.py — CuPy GPU example (requires matching CUDA/CuPy)
- unit_test_manufactured.py — manufactured-solution harness and a small smoke test
- tests/test_manufactured.py — pytest wrapper for CI

Usage (quick):
1. Clone repo
   git clone https://github.com/aiKC91/pde-relaxation.git
   cd pde-relaxation

2. Create venv and install deps (CPU-only):
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

3. Run a sample spectral solver:
   python spectral_solver.py --nx 128 --ny 128 --dt 0.002 --tend 1.0

4. Run tests (pytest):
   pytest -q

Notes:
- The spectral solver requires periodic BC.
- The SciPy solver uses sparse factorization for implicit diffusion; for large grids consider multigrid or FFT-based solvers.
- GPU examples require CuPy matching your CUDA toolkit.
- FiPy example requires FiPy; FiPy may pull additional dependencies.

See individual scripts for more options and comments on numerics (RK2 backtrace, interpolation choices, diagnostics).
