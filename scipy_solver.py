#!/usr/bin/env python3
"""
scipy_solver.py (enhanced)
Adds RK3 option and CLI selection for backtrace method.
"""
import argparse
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import factorized
try:
    from scipy.ndimage import map_coordinates
except Exception:
    map_coordinates = None


def build_periodic_laplacian(nx, ny, dx, dy):
    ex = np.ones(nx)
    Tx = sp.diags([ex, -2*ex, ex], [-1, 0, 1], shape=(nx, nx), format='csr').tolil()
    Tx[0, -1] = 1.0
    Tx[-1, 0] = 1.0
    Tx = Tx.tocsr()
    ey = np.ones(ny)
    Ty = sp.diags([ey, -2*ey, ey], [-1, 0, 1], shape=(ny, ny), format='csr').tolil()
    Ty[0, -1] = 1.0
    Ty[-1, 0] = 1.0
    Ty = Ty.tocsr()
    Ix = sp.eye(nx, format='csr')
    Iy = sp.eye(ny, format='csr')
    L = sp.kron(Iy, Tx / dx**2, format='csr') + sp.kron(Ty / dy**2, Ix, format='csr')
    return L


def bicubic_periodic_interp(field, x_pos, y_pos, Lx, Ly):
    nx, ny = field.shape
    dx = Lx / nx
    dy = Ly / ny
    xi = (x_pos / dx) - 0.5
    yi = (y_pos / dy) - 0.5
    coords = np.vstack([xi.ravel(), yi.ravel()])
    if map_coordinates is not None:
        sampled = map_coordinates(field, coords, order=3, mode='wrap')
    else:
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
        sampled = ((1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11).ravel()
    return sampled.reshape(field.shape)


def compute_curl_magnitude(u_x, u_y, dx, dy):
    dvy_dx = (np.roll(u_y, -1, axis=0) - np.roll(u_y, 1, axis=0)) / (2*dx)
    dux_dy = (np.roll(u_x, -1, axis=1) - np.roll(u_x, 1, axis=1)) / (2*dy)
    return np.abs(dvy_dx - dux_dy)


def integrate_backward_rk(X, Y, dt, vel_func, t, method='rk2'):
    h = -dt
    if method == 'euler':
        u_x, u_y = vel_func(X, Y, t)
        x0 = (X + h * u_x) % (X.max() + (X[1,0]-X[0,0]))
        y0 = (Y + h * u_y) % (Y.max() + (Y[0,1]-Y[0,0]))
        return x0, y0
    if method == 'rk2':
        u1x, u1y = vel_func(X, Y, t)
        X1 = X + 0.5 * h * u1x
        Y1 = Y + 0.5 * h * u1y
        u2x, u2y = vel_func(X1, Y1, t + 0.5*h)
        x0 = (X + h * u2x) % (X.max() + (X[1,0]-X[0,0]))
        y0 = (Y + h * u2y) % (Y.max() + (Y[0,1]-Y[0,0]))
        return x0, y0
    if method == 'rk3':
        u1x, u1y = vel_func(X, Y, t)
        X2 = X + 0.5 * h * u1x
        Y2 = Y + 0.5 * h * u1y
        u2x, u2y = vel_func(X2, Y2, t + 0.5*h)
        X3 = X + h * (-u1x + 2*u2x)
        Y3 = Y + h * (-u1y + 2*u2y)
        u3x, u3y = vel_func(X3, Y3, t + h)
        x0 = (X + h * (u1x / 6.0 + 2.0/3.0 * u2x + u3x / 6.0)) % (X.max() + (X[1,0]-X[0,0]))
        y0 = (Y + h * (u1y / 6.0 + 2.0/3.0 * u2y + u3y / 6.0)) % (Y.max() + (Y[0,1]-Y[0,0]))
        return x0, y0
    raise ValueError('Unknown method')


def default_velocity(X, Y, t):
    omega = 2.0 * np.pi
    xc, yc = 0.5, 0.5
    u_x = -omega * (Y - yc)
    u_y =  omega * (X - xc)
    return u_x, u_y


def run_solver(nx=128, ny=128, Lx=1.0, Ly=1.0, dt=0.002, t_end=0.5,
               alpha=1e-3, beta=5.0, gamma=1.0, rk='rk2'):
    dx = Lx / nx
    dy = Ly / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    Psi = np.exp(-((X-0.75)**2 + (Y-0.5)**2) / 0.005)
    F = np.exp(-((X-0.25)**2 + (Y-0.5)**2) / 0.02)

    L = build_periodic_laplacian(nx, ny, dx, dy)
    A = sp.eye(nx*ny, format='csr') - dt * alpha * L
    solve_implicit = factorized(A.tocsc())

    t = 0.0
    nsteps = int(np.ceil(t_end / dt))

    for step in range(nsteps):
        u_x, u_y = default_velocity(X, Y, t)
        x0, y0 = integrate_backward_rk(X, Y, dt, default_velocity, t, method=rk)
        Psi_adv = bicubic_periodic_interp(Psi, x0, y0, Lx, Ly)
        curl_mag = compute_curl_magnitude(u_x, u_y, dx, dy)
        S = -beta * Psi_adv * curl_mag + gamma * F
        rhs = (Psi_adv + dt * S).ravel()
        Psi = solve_implicit(rhs).reshape((nx, ny))
        t += dt
    return Psi


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--dt', type=float, default=0.002)
    p.add_argument('--tend', type=float, default=0.5)
    p.add_argument('--rk', choices=['euler','rk2','rk3'], default='rk2')
    args = p.parse_args()
    psi = run_solver(nx=args.nx, ny=args.ny, dt=args.dt, t_end=args.tend, rk=args.rk)
    print('Done SciPy run. max Psi =', np.max(psi))
