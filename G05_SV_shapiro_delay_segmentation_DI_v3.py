#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G05_SV_shapiro_delay_segmentation_DI_v3

Fixes:
- Vectorized midpoint integral (fast).
- DI-knife noise uses direct Gaussian draws with the correct scaling,
  avoiding any (samples,N) arrays.

Parts:
1) Shapiro-type delay Δt = ∫ (1/T^2 - 1) dx / c, check refinement stability in N.
2) DI-knife: forbidden micro-noise gives std ~ 1/sqrt(N); safe global noise gives std ~ const.
"""

import math
import numpy as np

# SI constants
c = 299792458.0
G = 6.67430e-11

# Sun
M_sun = 1.98892e30
R_sun = 6.9634e8
AU = 1.495978707e11


def lapse_from_point_mass(M: float, r: np.ndarray) -> np.ndarray:
    Phi = -G * M / r
    arg = 1.0 + 2.0 * Phi / (c * c)
    return np.sqrt(np.maximum(arg, 1e-30))


def shapiro_delay_integral_midpoint_vec(N: int, L: float, b: float, M: float) -> float:
    dx = (2.0 * L) / N
    x = -L + (np.arange(N, dtype=float) + 0.5) * dx
    r = np.sqrt(x * x + b * b)
    T = lapse_from_point_mass(M, r)
    integrand = (1.0 / (T * T) - 1.0)
    return float(np.sum(integrand) * dx / c)


def shapiro_delay_GR_approx(L: float, b: float, M: float) -> float:
    r1 = math.sqrt(L * L + b * b)
    r2 = r1
    return (2.0 * G * M / (c ** 3)) * math.log((4.0 * r1 * r2) / (b * b))


def fit_slope_loglog(x, y):
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(intercept)


def main() -> int:
    print("=== G05_SV_shapiro_delay_segmentation_DI_v3 ===")

    # Geometry
    L = AU
    b = 1.1 * R_sun
    print(f"L = {L:.3e} m (~1 AU)")
    print(f"b = {b:.3e} m (~1.1 R_sun)")
    print()

    # Part 1: integral stability
    N_LIST_INT = [256, 512, 1024, 2048, 4096, 8192]
    print("PART 1: Shapiro-type delay integral (vectorized midpoint)")
    print("     N |  Δt_CCM [μs] |   step diff [μs]")
    vals = []
    for N in N_LIST_INT:
        dt = shapiro_delay_integral_midpoint_vec(N=N, L=L, b=b, M=M_sun)
        vals.append(dt)

    for i, N in enumerate(N_LIST_INT):
        dt_us = vals[i] * 1e6
        if i == 0:
            print(f"{N:6d} | {dt_us:12.6f} | {'-':>14}")
        else:
            diff_us = (vals[i] - vals[i-1]) * 1e6
            print(f"{N:6d} | {dt_us:12.6f} | {diff_us:14.6f}")

    last_rel = abs(vals[-1] - vals[-2]) / abs(vals[-1])
    print()
    print("CHECK 1A: refinement stability (last step)")
    print(f"  relative change = {last_rel:.6e}")
    ok_1A = last_rel < 5e-6
    print(f"  RESULT: {'PASS' if ok_1A else 'FAIL'}")
    print()

    dt_GR = shapiro_delay_GR_approx(L=L, b=b, M=M_sun)
    err_GR = abs(vals[-1] - dt_GR) / abs(dt_GR)
    print("CHECK 1B: correspondence vs GR Shapiro approx")
    print(f"  Δt_CCM(N=max) = {vals[-1]*1e6:.6f} μs")
    print(f"  Δt_GR_approx  = {dt_GR*1e6:.6f} μs")
    print(f"  relative error = {err_GR:.6e}")
    ok_1B = err_GR < 5e-3
    print(f"  RESULT: {'PASS' if ok_1B else 'FAIL'}")
    print()

    # Part 2: DI-knife with noise, no huge arrays
    print("PART 2: DI-knife (segmentation dependence) on noisy 'delay fact'")
    seed = 1
    rng = np.random.default_rng(seed)

    # Keep this modest; you don't need 200k for slope detection
    samples = 80_000
    sigma0_us = 1.0
    sigma0 = sigma0_us * 1e-6

    N_LIST_NOISE = [256, 512, 1024, 2048, 4096, 8192]

    print(f"seed={seed}, samples={samples}, sigma0={sigma0_us:.2f} μs")
    print()

    # Model A: forbidden micro-noise => std ~ 1/sqrt(N)
    print("MODEL A: micro-noise sum (FORBIDDEN)")
    print("     N |   std[μs] | std*sqrt(N)[μs]")
    stds_A = []
    for N in N_LIST_NOISE:
        # equivalent to sum of N iid N(0, (sigma0/N)^2): std = sigma0/sqrt(N)
        noise = rng.normal(loc=0.0, scale=sigma0 / math.sqrt(N), size=samples)
        std = float(np.std(noise, ddof=0)) * 1e6
        stds_A.append(std)
        print(f"{N:6d} | {std:9.6f} | {std*math.sqrt(N):16.6f}")

    slope_A, _ = fit_slope_loglog(N_LIST_NOISE, stds_A)
    print(f"slope log(std) vs log(N) = {slope_A:+.4f}  (expected ~ -0.5)")
    ok_A = abs(slope_A + 0.5) < 0.04
    print(f"RESULT: {'PASS' if ok_A else 'FAIL'} (segmentation dependence detected)")
    print()

    # Model B: safe global noise => std independent of N
    print("MODEL B: global noise (DI-safe control)")
    print("     N |   std[μs]")
    stds_B = []
    for N in N_LIST_NOISE:
        noise = rng.normal(loc=0.0, scale=sigma0, size=samples)
        std = float(np.std(noise, ddof=0)) * 1e6
        stds_B.append(std)
        print(f"{N:6d} | {std:9.6f}")

    slope_B, _ = fit_slope_loglog(N_LIST_NOISE, stds_B)
    print(f"slope log(std) vs log(N) = {slope_B:+.4f}  (expected ~ 0)")
    ok_B = abs(slope_B) < 0.07
    print(f"RESULT: {'PASS' if ok_B else 'FAIL'} (control segmentation-invariant)")
    print()

    overall = ok_1A and ok_1B and ok_A and ok_B
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    # In notebooks you may want main() instead of SystemExit
    raise SystemExit(main())