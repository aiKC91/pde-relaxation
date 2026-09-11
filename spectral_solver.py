#!/usr/bin/env python3
"""
spectral_solver.py
Semi-Lagrangian + spectral implicit diffusion (FFTs) for periodic domains.
Recommended for fast periodic-domain runs.
"""
import argparse
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def default_velocity(X, Y, t, params):
    amp = params.get('amp', 1.0)
    kx = 2*np.pi
    ky = 2*np.pi
    tshift = np.cos(params.get('omega_t', 2.0)*t)
    u_x =  amp *  np.sin(kx*X) * np.cos(ky*Y) * tshift
    u_y = -amp *  np.cos(kx*X) * np.sin(ky*Y) * tshift
    return u_x, u_y


def bilinear_periodic_interp(field, x_pos, y_pos, Lx, Ly):
    nx, ny = field.shape
    dx = Lx / nx
    dy = Ly / ny
    xi = (x_pos / dx) - 0.5
    yi = (y_pos / dy) - 0.5
    i0 = np.floor(xi).astype(int) % nx
    j0 = np.floor(yi).astype(int) % ny
    tx = xi - np.floor(xi)
    ty = yi - np.floor(yi)
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]
    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


def compute_curl_magnitude(u_x, u_y, dx, dy):
    dvy_dx = (np.roll(u_y, -1, axis=0) - np.roll(u_y, 1, axis=0)) / (2*dx)
    dux_dy = (np.roll(u_x, -1, axis=1) - np.roll(u_x, 1, axis=1)) / (2*dy)
    curl_z = dvy_dx - dux_dy
    return np.abs(curl_z)


def spectral_implicit_solve(rhs, dt, alpha, k2):
    rhs_hat = fft2(rhs)
    denom = 1.0 + dt * alpha * k2
    psi_hat = rhs_hat / denom
    psi = np.real(ifft2(psi_hat))
    return psi


def run_simulation(nx=128, ny=128, Lx=1.0, Ly=1.0, dt=0.002, t_end=1.0,
                   alpha=1e-3, beta=5.0, gamma=1.0, plot=False):
    dx = Lx / nx
    dy = Ly / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    Psi = np.exp(-((X-0.75)**2 + (Y-0.5)**2) / 0.005)
    F = np.exp(-((X-0.25)**2 + (Y-0.5)**2) / 0.02)

    kx = 2*np.pi * fftfreq(nx, d=dx)
    ky = 2*np.pi * fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k2 = (KX**2 + KY**2)

    t = 0.0
    nsteps = int(np.ceil(t_end / dt))

    for step in range(nsteps):
        u_x, u_y = default_velocity(X, Y, t, {'omega_t':2.0, 'amp':1.0})
        x0 = (X - dt * u_x) % Lx
        y0 = (Y - dt * u_y) % Ly
        Psi_adv = bilinear_periodic_interp(Psi, x0, y0, Lx, Ly)
        curl_mag = compute_curl_magnitude(u_x, u_y, dx, dy)
        S = -beta * Psi_adv * curl_mag + gamma * F
        rhs = Psi_adv + dt * S
        Psi = spectral_implicit_solve(rhs, dt, alpha, k2)
        t += dt
    return Psi

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--dt', type=float, default=0.002)
    p.add_argument('--tend', type=float, default=1.0)
    args = p.parse_args()
    psi = run_simulation(nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.tend)
    print('Done spectral run. max Psi =', np.max(psi))
