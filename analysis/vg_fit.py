#!/usr/bin/env python3
"""
Van Genuchten fitting and simple hydraulic conductivity proxy utilities.

This module provides:
  - vg_theta(psi, theta_s, theta_r, alpha, n):
      Van Genuchten SWCC θ(ψ) with m = 1 - 1/n
  - fit_vg_alpha_n(psi, theta, theta_s, theta_r):
      Robust grid+refine fitting of (alpha, n) with fixed θs, θr
  - mualem_kr_from_theta(theta, theta_s, theta_r, l=0.5):
      Minimal relative conductivity proxy K_r(ψ) based on Se^l

Intended for:
  - Simulation usability analysis of SWCC predictors (Observed / GB / PINN)
  - Showing that monotone PINN curves are fit-stable and yield consistent K_r(ψ)
"""

import numpy as np


def vg_theta(psi, theta_s, theta_r, alpha, n):
    """
    Van Genuchten SWCC:

        Se(ψ) = (θ - θ_r) / (θ_s - θ_r)
              = [1 + (α ψ)^n]^(-m),  m = 1 - 1/n
        θ(ψ)  = θ_r + (θ_s - θ_r) * Se(ψ)

    Args:
        psi:      suction array (kPa), shape [P]
        theta_s:  saturated water content (scalar)
        theta_r:  residual water content (scalar)
        alpha:    VG parameter α (1/kPa)
        n:        VG parameter n (> 1)

    Returns:
        theta: array [P] of θ(ψ)
    """
    psi = np.asarray(psi, dtype=float)
    theta_s = float(theta_s)
    theta_r = float(theta_r)
    alpha = float(alpha)
    n = max(float(n), 1.0001)  # ensure n > 1

    m = 1.0 - 1.0 / n
    # Avoid overflow / underflow
    psi_safe = np.maximum(psi, 1e-8)
    Se = (1.0 + (alpha * psi_safe) ** n) ** (-m)
    return theta_r + (theta_s - theta_r) * Se


def fit_vg_alpha_n(
    psi,
    theta,
    theta_s,
    theta_r,
    bounds=((1e-8, 1.01), (1e2, 20.0)),
    n_grid: int = 30,
    refine_steps: int = 40,
):
    """
    Fit α and n with fixed θs, θr using a robust grid + coordinate-descent refinement.

    This avoids external curve-fitting dependencies and is stable for noisy data.

    Args:
        psi:       suction array [P] (kPa)
        theta:     water content array [P]
        theta_s:   known θs for this sample (scalar)
        theta_r:   known θr for this sample (scalar)
        bounds:    ((alpha_min, n_min), (alpha_max, n_max))
        n_grid:    grid resolution per dimension for coarse search
        refine_steps: number of local refinement iterations

    Returns:
        alpha:      fitted α
        n:          fitted n
        rmse:       RMSE between θ and θ_VG_fit
        theta_fit:  fitted θ(ψ) array [P]
    """
    psi = np.asarray(psi, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # Basic sanity: need enough finite points
    mask = np.isfinite(psi) & np.isfinite(theta)
    if np.sum(mask) < 5:
        return np.nan, np.nan, np.inf, np.full_like(theta, np.nan)

    psi = psi[mask]
    theta = theta[mask]

    # Effective saturation from observed curve
    denom = max(theta_s - theta_r, 1e-6)
    Se_obs = np.clip((theta - theta_r) / denom, 1e-6, 1.0 - 1e-6)

    # Loss function over (alpha, n): MSE in Se-space
    def loss(alpha, n_val):
        n_val = max(float(n_val), 1.0001)
        m_val = 1.0 - 1.0 / n_val
        psi_safe = np.maximum(psi, 1e-8)
        Se = (1.0 + (alpha * psi_safe) ** n_val) ** (-m_val)
        return float(np.mean((Se - Se_obs) ** 2))

    (alpha_min, n_min), (alpha_max, n_max) = bounds

    # Coarse grid initialization (log for alpha, linear for n)
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_grid)
    ns = np.linspace(n_min, n_max, n_grid)

    best_alpha, best_n, best_L = None, None, np.inf
    for a in alphas:
        for n_val in ns:
            L = loss(a, n_val)
            if L < best_L:
                best_alpha, best_n, best_L = a, n_val, L

    alpha = best_alpha
    n_val = best_n

    # Local refinement by simple coordinate descent
    for _ in range(refine_steps):
        # refine alpha
        a_candidates = alpha * np.array([0.6, 0.8, 1.0, 1.25, 1.6], dtype=float)
        a_candidates = np.clip(a_candidates, alpha_min, alpha_max)
        Ls = [loss(a, n_val) for a in a_candidates]
        alpha = float(a_candidates[int(np.argmin(Ls))])

        # refine n
        n_candidates = n_val + np.array([-0.6, -0.3, 0.0, 0.3, 0.6], dtype=float)
        n_candidates = np.clip(n_candidates, n_min, n_max)
        Ls = [loss(alpha, nn) for nn in n_candidates]
        n_val = float(n_candidates[int(np.argmin(Ls))])

    # Final fitted curve and RMSE
    theta_fit = vg_theta(psi, theta_s, theta_r, alpha, n_val)
    rmse = float(np.sqrt(np.mean((theta_fit - theta) ** 2)))
    return alpha, n_val, rmse, theta_fit


def mualem_kr_from_theta(theta, theta_s, theta_r, l: float = 0.5):
    """
    Minimal relative hydraulic conductivity proxy from θ(ψ):

        Se = (θ - θ_r) / (θ_s - θ_r)
        K_r ∝ Se^l

    This does NOT implement full Mualem-VG (which needs VG m and n),
    but is sufficient to check monotonicity and qualitative behavior of K_r(ψ).

    Args:
        theta:   θ(ψ) array [P]
        theta_s: θs (scalar)
        theta_r: θr (scalar)
        l:       pore-connectivity exponent (default 0.5)

    Returns:
        K_r: relative conductivity array [P], in [0, 1]
    """
    theta = np.asarray(theta, dtype=float)
    denom = max(theta_s - theta_r, 1e-6)
    Se = np.clip((theta - theta_r) / denom, 1e-8, 1.0)
    return Se ** float(l)

