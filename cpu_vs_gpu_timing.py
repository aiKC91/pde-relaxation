#!/usr/bin/env python3
"""
cpu_vs_gpu_timing.py
Simple script that imports the GPU and CPU solvers and reports timings.
"""
from gpu_solver import run_cpu_spectral, run_gpu_solver


def main():
    nx = 256
    ny = 256
    dt = 0.002
    tend = 0.5
    print('CPU run...')
    _, t_cpu = run_cpu_spectral(nx=nx, ny=ny, dt=dt, t_end=tend)
    print('CPU elapsed:', t_cpu)
    try:
        _, t_gpu = run_gpu_solver(nx=nx, ny=ny, dt=dt, t_end=tend)
        print('GPU elapsed:', t_gpu)
    except Exception as e:
        print('GPU run failed or CuPy missing:', e)

if __name__ == '__main__':
    main()
