# WENO scaffold

This file describes a planned WENO interpolation module for semi-Lagrangian advection.

Contents/plan:
- Implement 2D dimension-by-dimension WENO5 reconstruction for cell-centered fields.
- Provide monotonicity-preserving limiter option.
- API: weno_interp(field, x_pos, y_pos, order=5, mode='periodic') -> interpolated values

References:
- Shu, C.-W. (1998). Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws.
- Jiang & Shu (1996), WENO schemes.

TODO: full implementation is nontrivial; currently this repo contains a scaffold and references. If you want a full WENO implementation added, reply and I will implement it.
