#!/usr/bin/env python3
"""
spectral_solver.py (enhanced)
Semi-Lagrangian + spectral implicit diffusion for periodic domains.
Supports RK backtrace: euler, rk2, rk3
"""
import argparse
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def default_velocity(X, Y, t, params=None):
    amp = 1.0 if params is None else params.get('amp', 1.0)
    kx = 2*np.pi
    ky = 2*np.pi
    tshift = np.cos(2.0*t)
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


def backtrace_characteristics(X, Y, dt, vel_func, t, method='rk2', params=None):
    # integrate backward for time dt using chosen Runge-Kutta method
    # we integrate ODE dx/ds = u(x,s) from s=t to s=t-dt with step h = -dt
    h = -dt
    if method == 'euler':
        u_x, u_y = vel_func(X, Y, t, params)
        x0 = (X + h * u_x) % (X.max() + (X[1,0]-X[0,0]))
        y0 = (Y + h * u_y) % (Y.max() + (Y[0,1]-Y[0,0]))
        return x0, y0
    if method == 'rk2':
        u1x, u1y = vel_func(X, Y, t, params)
        X1 = X + 0.5 * h * u1x
        Y1 = Y + 0.5 * h * u1y
        u2x, u2y = vel_func(X1, Y1, t + 0.5*h, params)
        x0 = (X + h * u2x) % (X.max() + (X[1,0]-X[0,0]))
        y0 = (Y + h * u2y) % (Y.max() + (Y[0,1]-Y[0,0]))
        return x0, y0
    if method == 'rk3':
        # classical RK3 (Kutta) with step h
        u1x, u1y = vel_func(X, Y, t, params)
        X2 = X + 0.5 * h * u1x
        Y2 = Y + 0.5 * h * u1y
        u2x, u2y = vel_func(X2, Y2, t + 0.5*h, params)
        X3 = X + h * (-u1x + 2*u2x)
        Y3 = Y + h * (-u1y + 2*u2y)
        u3x, u3y = vel_func(X3, Y3, t + h, params)
        x0 = (X + h * (u1x / 6.0 + 2.0/3.0 * u2x + u3x / 6.0)) % (X.max() + (X[1,0]-X[0,0]))
        y0 = (Y + h * (u1y / 6.0 + 2.0/3.0 * u2y + u3y / 6.0)) % (Y.max() + (Y[0,1]-Y[0,0]))
        return x0, y0
    raise ValueError('Unknown RK method')


def run_simulation(nx=128, ny=128, Lx=1.0, Ly=1.0, dt=0.002, t_end=1.0,
                   alpha=1e-3, beta=5.0, gamma=1.0, rk='rk2'):
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
        u_x, u_y = default_velocity(X, Y, t, None)
        x0, y0 = backtrace_characteristics(X, Y, dt, default_velocity, t, method=rk)
        Psi_adv = bilinear_periodic_interp(Psi, x0, y0, Lx, Ly)
        curl_mag = compute_curl_magnitude(u_x, u_y, dx, dy)
        S = -beta * Psi_adv * curl_mag + gamma * F
        rhs = Psi_adv + dt * S
        Psi = spectral_implicit_solve(rhs, dt, alpha, k2)
        t += dt
    return Psi

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--dt', type=float, default=0.002)
    p.add_argument('--tend', type=float, default=1.0)
    p.add_argument('--rk', choices=['euler','rk2','rk3'], default='rk2')
    args = p.parse_args()
    psi = run_simulation(nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.tend, rk=args.rk)
    print('Done spectral run. max Psi =', np.max(psi))
