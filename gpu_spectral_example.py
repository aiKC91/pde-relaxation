#!/usr/bin/env python3
"""
gpu_spectral_example.py
CuPy-based spectral implicit diffusion + semi-Lagrangian example.
Requires CuPy installed for your CUDA version.
"""
try:
    import cupy as cp
    from cupy.fft import fft2 as cfft2, ifft2 as cifft2
    from cupyx.scipy.ndimage import map_coordinates as cmap_coordinates
except Exception:
    cp = None
import numpy as np

if cp is None:
    print('CuPy not installed; this is an example. Install cupy for GPU runs.')

else:
    nx, ny = 256, 256
    Lx, Ly = 1.0, 1.0
    dx = Lx / nx
    dy = Ly / ny
    x = (cp.arange(nx) + 0.5) * dx
    y = (cp.arange(ny) + 0.5) * dy
    X, Y = cp.meshgrid(x, y, indexing='ij')

    Psi = cp.exp(-((X-0.75)**2 + (Y-0.5)**2) / 0.005)
    omega = 2*cp.pi
    xc, yc = 0.5*Lx, 0.5*Ly
    u_x = -omega * (Y - yc)
    u_y =  omega * (X - xc)

    kx = 2*cp.pi * cp.fft.fftfreq(nx, d=dx)
    ky = 2*cp.pi * cp.fft.fftfreq(ny, d=dy)
    KX, KY = cp.meshgrid(kx, ky, indexing='ij')
    k2 = (KX**2 + KY**2)

    dt = 0.002
    alpha = 1e-3
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
    Psi_new = cp.real(cifft2(psi_hat))
    print('GPU step done. max Psi_new =', float(Psi_new.max()))
