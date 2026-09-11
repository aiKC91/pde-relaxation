# Pull Request: Feature/enhancements -> ETERNAFX

Title: Add RK3 backtrace, GPU solver & timing, packaging, tests, and scaffolds

Summary
-------
This PR implements the recommended enhancements to the pde-relaxation project on branch `feature/enhancements`. Key changes include:

- Solvers
  - spectral_solver.py: added RK backtraces (euler, rk2, rk3) with CLI option `--rk`.
  - scipy_solver.py: RK selection (euler, rk2, rk3), bicubic interpolation, CLI option `--rk`.
  - gpu_solver.py: full end-to-end CuPy spectral solver that keeps arrays on-device and reports timing.
  - cpu_vs_gpu_timing.py: timing wrapper to compare CPU vs GPU runs.

- Packaging & reproducibility
  - Dockerfile for a CPU-based image
  - environment.yml (conda) and requirements.txt

- Tests & CI
  - Added pytest tests under `tests/` (RK smoke tests and manufactured-solution smoke test)
  - CI workflow updated to run pytest and a black formatting check on pushes/PRs to ETERNAFX and feature/enhancements branches

- Scaffolds & docs
  - WENO_SCAFFOLD.md: notes and references for future WENO interpolation implementation
  - mpi_spectral_scaffold.py: MPI slab-decomposition scaffold for distributed spectral solves
  - tools/black_check.py: helper used in CI

Motivation
---------
These changes improve accuracy (RK3 backtrace), provide a GPU implementation and timing harness for performance comparison, and add reproducible environment artifacts and CI tests to make development and review easier.

Files changed (high level)
-------------------------
- spectral_solver.py (modified)
- scipy_solver.py (modified)
- gpu_solver.py (new)
- cpu_vs_gpu_timing.py (new)
- Dockerfile (new)
- environment.yml (new)
- WENO_SCAFFOLD.md (new)
- mpi_spectral_scaffold.py (new)
- tests/* (new)
- .github/workflows/ci.yml (updated)

Checklist for review
--------------------
- [ ] Verify RK options work: run `spectral_solver.py --rk rk3` and `scipy_solver.py --rk rk3` on a small grid.
- [ ] Run pytest locally: `pytest -q` (ensure black is installed or run `python -m black .`)
- [ ] Run GPU solver if you have CuPy: `python gpu_solver.py --nx 256 --ny 256 --tend 0.5`.
- [ ] Build Docker image (optional): `docker build -t pde-relaxation:cpu .` and run.
- [ ] Inspect WENO_SCAFFOLD.md and MPI scaffold for future implementation tasks.
- [ ] Confirm CI passes on the PR; if not, check formatting via `black`.

Testing instructions
--------------------
1. Setup environment (venv or conda) and install dependencies from `requirements.txt` or `environment.yml`.
2. Run a small spectral test:
   python spectral_solver.py --nx 64 --ny 64 --dt 0.005 --tend 0.05 --rk rk3
3. Run SciPy solver test (RK3):
   python scipy_solver.py --nx 64 --ny 64 --dt 0.005 --tend 0.05 --rk rk3
4. Run pytest: pytest -q

Notes / TODO
------------
- WENO is scaffolded only — full implementation to be added in a follow-up PR if desired.
- MPI scaffold outlines domain decomposition; full parallel FFT support (pyFFTW/p3dfft) can be added later.

If you'd like, I can create the PR on GitHub for you using the repository's web UI or via the gh/curl command shown below.

Commands to create PR (use locally or in CI):
- Using GitHub CLI (recommended):
  gh pr create --base ETERNAFX --head feature/enhancements --title "Add RK3 backtrace, GPU solver & timing, packaging, tests" --body-file PR_BODY.md

- Using curl (replace GITHUB_TOKEN):
  curl -X POST -H "Authorization: token $GITHUB_TOKEN" -d '{"title":"Add RK3 backtrace, GPU solver & timing, packaging, tests","head":"feature/enhancements","base":"ETERNAFX","body":"(see PR_BODY.md)"}' https://api.github.com/repos/aiKC91/pde-relaxation/pulls

