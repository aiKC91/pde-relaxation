# Reddit posts (tailored)

r/MachineLearning (focus: data generation & ML use cases)

Title: Fast PDE relaxation for dataset generation (GPU acc.) — useful for PINNs / Neural Operators

Text:
Hey ML folks — I made a PDE relaxation solver with optional GPU acceleration (CuPy). It’s lightweight, reproducible, and can generate large-scale training data much faster than typical CPU code. It comes with a demo notebook and benchmark scripts. Would love feedback on data quality and integration with PINNs.

Repo: https://github.com/aiKC91/pde-relaxation
Notebook: docs/getting_started.ipynb
Benchmarks: See README for template

r/Physics / r/ComputationalPhysics (focus: numerical methods)

Title: Practical GPU-accelerated PDE relaxation: RK integrators + CuPy

Text:
I implemented a relaxation solver that leverages CuPy for GPUs and provides reproducible CPU fallbacks and benchmarks. I’m looking for feedback on numerical stability and boundary condition support. Open to collaborators for adding WENO / MPI examples.

Repo link and details in README.
