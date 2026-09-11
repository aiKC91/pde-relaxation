#!/usr/bin/env python3
"""
gpu_solver.py
Full GPU spectral semi-Lagrangian solver with timing.
Requires CuPy matching your CUDA version.
"""
import time
import numpy as np
try:
    import cupy as cp
    from cupy.fft import fft2 as cfft2, ifft2 as cifft2
    from cupyx.scipy.ndimage import map_coordinates as cmap_coordinates
except Exception:
    cp = None


def run_gpu_solver(nx=256, ny=256, Lx=1.0, Ly=1.0, dt=0.002, t_end=0.5, alpha=1e-3, steps=None):
    if cp is None:
        raise RuntimeError('CuPy not available')
    dx = Lx / nx
    dy = Ly / ny
    x = (cp.arange(nx) + 0.5) * dx
    y = (cp.arange(ny) + 0.5) * dy
    X, Y = cp.meshgrid(x, y, indexing='ij')
    Psi = cp.exp(-((X-0.75)**2 + (Y-0.5)**2) / 0.005)
    omega = 2*cp.pi
    xc, yc = 0.5*Lx, 0.5*Ly
    kx = 2*cp.pi * cp.fft.fftfreq(nx, d=dx)
    ky = 2*cp.pi * cp.fft.fftfreq(ny, d=dy)
    KX, KY = cp.meshgrid(kx, ky, indexing='ij')
    k2 = (KX**2 + KY**2)
    t = 0.0
    nsteps = int(np.ceil(t_end / dt)) if steps is None else steps
    start = time.time()
    for step in range(nsteps):
        u_x = -omega * (Y - yc)
        u_y =  omega * (X - xc)
        x0 = (X - dt * u_x) % Lx
        y0 = (Y - dt * u_y) % Ly
        xi = (x0 / dx) - 0.5
        yi = (y0 / dy) - 0.5
        coords = cp.stack([xi.ravel(), yi.ravel()])
        Psi_adv_flat = cmap_coordinates(Psi, coords, order=3, mode='wrap')
        Psi_adv = Psi_adv_flat.reshape((nx, ny))
        rhs_hat = cfft2(Psi_adv)
        denom = 1.0 + dt * alpha * k2
        psi_hat = rhs_hat / denom
        Psi = cp.real(cifft2(psi_hat))
        t += dt
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    return Psi, elapsed


def run_cpu_spectral(nx=256, ny=256, Lx=1.0, Ly=1.0, dt=0.002, t_end=0.5):
    from numpy.fft import fft2, ifft2, fftfreq
    dx = Lx / nx
    dy = Ly / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    Psi = np.exp(-((X-0.75)**2 + (Y-0.5)**2) / 0.005)
    omega = 2*np.pi
    xc, yc = 0.5*Lx, 0.5*Ly
    kx = 2*np.pi * fftfreq(nx, d=dx)
    ky = 2*np.pi * fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k2 = (KX**2 + KY**2)
    t = 0.0
    nsteps = int(np.ceil(t_end / dt))
    start = time.time()
    for step in range(nsteps):
        u_x = -omega * (Y - yc)
        u_y =  omega * (X - xc)
        x0 = (X - dt * u_x) % Lx
        y0 = (Y - dt * u_y) % Ly
        xi = (x0 / dx) - 0.5
        yi = (y0 / dy) - 0.5
        # simple bilinear interp
        i0 = np.floor(xi).astype(int) % nx
        j0 = np.floor(yi).astype(int) % ny
        tx = xi - np.floor(xi)
        ty = yi - np.floor(yi)
        i1 = (i0 + 1) % nx
        j1 = (j0 + 1) % ny
        f00 = Psi[i0, j0]
        f10 = Psi[i1, j0]
        f01 = Psi[i0, j1]
        f11 = Psi[i1, j1]
        Psi_adv = (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11
        rhs_hat = fft2(Psi_adv)
        psi_hat = rhs_hat / (1.0 + dt * 1e-3 * k2)
        Psi = np.real(ifft2(psi_hat))
        t += dt
    elapsed = time.time() - start
    return Psi, elapsed

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=256)
    p.add_argument('--ny', type=int, default=256)
    p.add_argument('--dt', type=float, default=0.002)
    p.add_argument('--tend', type=float, default=0.5)
    args = p.parse_args()
    print('Running CPU spectral...')
    _, t_cpu = run_cpu_spectral(nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.tend)
    print('CPU time:', t_cpu)
    if cp is not None:
        print('Running GPU spectral...')
        _, t_gpu = run_gpu_solver(nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.tend)
        print('GPU time:', t_gpu)
    else:
        print('CuPy not installed; skip GPU run')
