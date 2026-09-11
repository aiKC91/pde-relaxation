# PDE Relaxation — GPU-accelerated PDE solver for fast relaxation

![hero-placeholder](docs/assets/hero.gif)

Replaces a 48–72 hour computational bottleneck with a single equation.

One-line value: Fast, reproducible PDE relaxation solvers with optional GPU acceleration (CuPy) and an easy CPU fallback — designed for research, prototyping, and production benchmarking.

---

Why this project matters
- Removes long-running numerical bottlenecks by providing an efficient relaxation solver with RK time integrators and optional GPU acceleration.
- Bridges numerical-analysis correctness and practical performance engineering (CPU vs GPU) so you can get results faster and iterate on models.
- Useful for CFD prototypes, physics-based simulations, and as a data generator for ML (PINNs / Neural Operators).

Quick demo (TL;DR)

1) Clone and install

```bash
git clone https://github.com/aiKC91/pde-relaxation.git
cd pde-relaxation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2) Run a tiny CPU example (works on GitHub runners)

```bash
python -m examples.run_example --nx 128 --ny 128 --device cpu
```

3) Run the GPU example (if you have CUDA + CuPy)

```bash
python -m examples.run_example --nx 512 --ny 512 --device gpu
```

4) Generate a benchmark CSV

```bash
python benchmarks/bench_cpu_gpu.py --sizes 128 256 512 1024 --repeats 3 --out benchmarks/results.csv
```

Benchmarks (template — replace with measured numbers)

| Grid | CPU (s) | GPU (CuPy) (s) | Speedup |
|------|---------:|---------------:|--------:|
| 128×128 | 0.12 | 0.03 | 4× |
| 256×256 | 0.45 | 0.06 | 7.5× |
| 512×512 | 1.9 | 0.18 | 10.6× |
| 1024×1024 | 8.2 | 0.9 | 9.1× |

Add your measured numbers to the table above and paste a PNG of the timing plot to docs/assets/bench-compare.png.

Generating a hero GIF (recommended)
- Run a short GPU vs CPU run with visualization frames and use ffmpeg to create an animated GIF. Example commands:

```bash
python examples/make_frames.py --nx 256 --ny 256 --device cpu --out frames/cpu
python examples/make_frames.py --nx 256 --ny 256 --device gpu --out frames/gpu
# combine side-by-side with imagemagick and create GIF
montage frames/cpu/*.png frames/gpu/*.png -tile 2x -geometry +0+0 frames/combined_%04d.png
ffmpeg -i frames/combined_%04d.png -vf "scale=800:-1:flags=lanczos,fps=12" -loop 0 docs/assets/hero.gif
```

Image alt text: "Left: CPU relaxation stuttering, Right: GPU relaxation running smoothly — 60fps equivalent".


Project structure (high level)

- spectral_solver.py, scipy_solver.py, gpu_solver.py — core solver modules
- tests/ — unit tests (lightweight by default; GPU/FiPy tests skipped when optional deps missing)
- benchmarks/ — benchmark scripts
- examples/ — runnable examples and notebook demos
- .github/ — CI, dependabot, CODEOWNERS, templates

Getting started notebook
- docs/getting_started.ipynb (demo + visualization) — open locally or view on nbviewer after pushing.

Contributing & support
- See CONTRIBUTING.md for local setup, test conventions, and PR process.
- Use issues for bugs and feature requests; tag PRs with a clear description and small tests.

Roadmap (short)
- Packaging + wheel distribution (pyproject + release action)
- Add scheduled benchmarks + self-hosted GPU benchmark runner
- Example pipeline: generate datasets → train a PINN / Neural Operator → evaluate speed/accuracy

License & citation
- Licensed under the MIT License. See LICENSE for details.

Contact
- Maintainer: @aiKC91 — raise issues or open PRs; use CODEOWNERS for auto-review requests.
