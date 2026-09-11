#!/usr/bin/env python3
"""
fipy_solver.py
FiPy finite-volume example for:
    dPsi/dt + u·∇Psi = alpha ∇²Psi - beta Psi |curl(u)| + gamma F

Requires FiPy installed.
"""
try:
    from fipy import PeriodicGrid2D, CellVariable, FaceVariable, TransientTerm, DiffusionTerm, ConvectionTerm, Viewer
except Exception:
    PeriodicGrid2D = None
import numpy as np

if PeriodicGrid2D is None:
    print('FiPy not available; this file is an example only.')

else:
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx = Lx / nx
    dy = Ly / ny
    mesh = PeriodicGrid2D(nx=nx, ny=ny, dx=dx, dy=dy)

    x = mesh.cellCenters[0].reshape((nx, ny))
    y = mesh.cellCenters[1].reshape((nx, ny))

    Psi0 = np.exp(-((x-0.75)**2 + (y-0.5)**2) / 0.005).ravel()
    phi = CellVariable(name='Psi', mesh=mesh, value=Psi0)

    F_cell = np.exp(-((x-0.25)**2 + (y-0.5)**2) / 0.02).ravel()

    alpha = 1e-3
    beta = 5.0
    gamma = 1.0
    dt = 0.002
    t_end = 0.2

    def u_cell(X, Y, t=0.0):
        u_x =  np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        u_y = -np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)
        return u_x, u_y

    def compute_face_velocity():
        face_x = mesh.faceCenters[0]
        face_y = mesh.faceCenters[1]
        ux_face, uy_face = u_cell(face_x, face_y)
        faceVel = FaceVariable(mesh=mesh, value=(ux_face, uy_face))
        return faceVel

    faceVel = compute_face_velocity()
    eq = TransientTerm() + ConvectionTerm(coeff=faceVel) - DiffusionTerm(coeff=alpha)

    t = 0.0
    viewer = None
    try:
        viewer = Viewer(vars=(phi,), datamin=0., datamax=1.)
    except Exception:
        pass

    while t < t_end:
        faceVel = compute_face_velocity()
        uxc, uyc = u_cell(mesh.cellCenters[0], mesh.cellCenters[1])
        ux = uxc.reshape((nx, ny))
        uy = uyc.reshape((nx, ny))
        dvy_dx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2*dx)
        dux_dy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2*dy)
        curl_mag = np.abs(dvy_dx - dux_dy).ravel()
        S = (-beta * phi.value * curl_mag + gamma * F_cell)
        eq = TransientTerm() + ConvectionTerm(coeff=faceVel) - DiffusionTerm(coeff=alpha)
        eq.solve(var=phi, dt=dt)
        phi.setValue(phi.value + dt * S)
        t += dt
        if viewer:
            viewer.plot()
    print('FiPy example done. final mass:', np.sum(phi.value) * dx * dy)
