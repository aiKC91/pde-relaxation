# Hacker News (Show HN) post draft

Title: Show HN: Solved a 48–72 hour PDE relaxation bottleneck with a single GPU-accelerated equation

Body (copy/paste):

Hi HN — I built a PDE relaxation solver that replaces long-running iterative workflows with a single, GPU-accelerated relaxation equation. It’s useful for prototype CFD, physics-based data generation, and rapid iteration when you need many simulation runs.

Why it matters
- Many researchers spend days waiting for relaxation to converge on large grids. This code demonstrates orders-of-magnitude speedups using CuPy and an RK integrator.
- Includes CPU fallback, benchmark scripts, and a small notebook.

Repo: https://github.com/aiKC91/pde-relaxation
Demo / Notebook: docs/getting_started.ipynb
Benchmarks: benchmarks/results.csv (example template)

If you’re into numerical methods, HPC, or physics-informed ML, I’d love feedback on accuracy, stability, and ways to integrate this into existing pipelines.

What I’d appreciate
- People to try it on their hardware and paste benchmark numbers
- Suggestions for better visualization of convergence and stability
- Help adding more boundary conditions and example datasets

Thanks — I’ll be monitoring and responding to comments.
