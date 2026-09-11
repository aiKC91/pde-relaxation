#!/usr/bin/env python3
"""
unit_test_manufactured.py
Manufactured solution and small-smoke test for convergence and CI.
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def analytic_solution(X, Y, t, kx, ky, lam):
    return np.exp(-lam * t) * np.sin(kx * X) * np.sin(ky * Y)


def compute_forcing(X, Y, t, u_x, u_y, kx, ky, lam, alpha, beta, gamma):
    psi = analytic_solution(X, Y, t, kx, ky, lam)
    dpt = -lam * psi
    dpx = kx * np.exp(-lam * t) * np.cos(kx * X) * np.sin(ky * Y)
    dpy = ky * np.exp(-lam * t) * np.sin(kx * X) * np.cos(ky * Y)
    adv = u_x * dpx + u_y * dpy
    lap = - (kx**2 + ky**2) * psi
    curl_mag = 0.0
    F = (dpt + adv - alpha * lap + beta * psi * curl_mag) / gamma
    return F


def spectral_solver_with_forcing(nx, ny, Lx, Ly, dt, t_end, alpha, beta, gamma, u_x_val, u_y_val, psi0, F_time_func):
    dx = Lx / nx
    dy = Ly / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    kx_v = 2*np.pi * fftfreq(nx, d=dx)
    ky_v = 2*np.pi * fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx_v, ky_v, indexing='ij')
    k2 = (KX**2 + KY**2)
    Psi = psi0.copy()
    t = 0.0
    nsteps = int(np.ceil(t_end / dt))
    for step in range(nsteps):
        x0 = (X - dt * u_x_val) % Lx
        y0 = (Y - dt * u_y_val) % Ly
        # simple bilinear interp for the manufactured test to keep dependency small
        xi = (x0 / dx) - 0.5
        yi = (y0 / dy) - 0.5
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
        F = F_time_func(X, Y, t)
        rhs = Psi_adv + dt * ( - beta * Psi_adv * 0.0 + gamma * F )
        rhs_hat = fft2(rhs)
        psi_hat = rhs_hat / (1.0 + dt * alpha * k2)
        Psi = np.real(ifft2(psi_hat))
        t += dt
    return Psi


def run_smoke_test():
    nx = 32
    ny = 32
    Lx = 1.0
    Ly = 1.0
    kx = 2*np.pi * 1
    ky = 2*np.pi * 1
    lam = 1.0
    alpha = 1e-3
    beta = 0.0
    gamma = 1.0
    u_x_val = 0.3
    u_y_val = 0.0
    t_end = 0.02
    dx = Lx / nx
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y, indexing='ij')
    psi0 = analytic_solution(X, Y, 0.0, kx, ky, lam)
    def F_time(Xc, Yc, t):
        return compute_forcing(Xc, Yc, t, u_x_val, u_y_val, kx, ky, lam, alpha, beta, gamma)
    Psi_num = spectral_solver_with_forcing(nx, ny, Lx, Ly, dt=0.0025, t_end=t_end, alpha=alpha, beta=beta, gamma=gamma, u_x_val=u_x_val, u_y_val=u_y_val, psi0=psi0, F_time_func=F_time)
    Psi_exact = analytic_solution(X, Y, t_end, kx, ky, lam)
    err = np.sqrt(np.sum((Psi_num - Psi_exact)**2) * (Lx/nx) * (Ly/ny))
    return err

if __name__ == '__main__':
    err = run_smoke_test()
    print('smoke L2 error =', err)
