#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T12_SV_DI_segmentation_knife

Single-file, single-run test.

Purpose
-------
Detect whether a claimed "physical fact" depends on segmentation N (a pure description choice).
If statistics change with N while mean is fixed, DI (Description Invariance) is violated.

Two templates:
  Model A: micro-sum along a path (FORBIDDEN template) -> std ~ 1/sqrt(N)
  Model B: global stop (DI-safe control template)      -> std ~ const

No external deps beyond numpy.
"""

import math
import numpy as np

def fit_slope_loglog(x, y):
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(intercept)

def simulate_model_A_micro_sum(rng, samples, N):
    # Sum of N iid exponentials with rate=N gives mean=1, std=1/sqrt(N)
    # Exp(scale=1/rate) => scale = 1/N
    x = rng.exponential(scale=1.0/N, size=(samples, N)).sum(axis=1)
    return x

def simulate_model_B_global_stop(rng, samples, N):
    # Global stop: draw ONE exponential with mean=1, independent of segmentation N
    # N is ignored by design.
    x = rng.exponential(scale=1.0, size=samples)
    return x

def main():
    seed = 1
    samples = 200_000
    N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    rng = np.random.default_rng(seed)

    print("=== T12_SV_DI_segmentation_knife ===")
    print(f"seed={seed}, samples={samples}")
    print(f"N_LIST = {N_LIST}")
    print()

    # ---------------- Model A ----------------
    print("MODEL A: micro-sum (FORBIDDEN template)")
    print("    N |     mean |      std | std*sqrt(N)")
    means_A, stds_A = [], []
    for N in N_LIST:
        x = simulate_model_A_micro_sum(rng, samples, N)
        mean = float(np.mean(x))
        std  = float(np.std(x, ddof=0))
        means_A.append(mean)
        stds_A.append(std)
        print(f"{N:5d} | {mean:8.4f} | {std:8.4f} | {std*math.sqrt(N):10.4f}")

    slope_A, _ = fit_slope_loglog(N_LIST, stds_A)
    print(f"slope log(std) vs log(N) = {slope_A:+.4f}  (expected ~ -0.5)")
    ok_A = abs(slope_A + 0.5) < 0.02
    print(f"RESULT: {'PASS' if ok_A else 'FAIL'} (segmentation dependence detected)")
    print()

    # ---------------- Model B ----------------
    print("MODEL B: global stop (DI-safe control template)")
    print("    N |     mean |      std")
    means_B, stds_B = [], []
    for N in N_LIST:
        x = simulate_model_B_global_stop(rng, samples, N)
        mean = float(np.mean(x))
        std  = float(np.std(x, ddof=0))
        means_B.append(mean)
        stds_B.append(std)
        print(f"{N:5d} | {mean:8.4f} | {std:8.4f}")

    slope_B, _ = fit_slope_loglog(N_LIST, stds_B)
    print(f"slope log(std) vs log(N) = {slope_B:+.4f}  (expected ~ 0)")
    ok_B = abs(slope_B) < 0.05
    print(f"RESULT: {'PASS' if ok_B else 'FAIL'} (control is segmentation-invariant)")
    print()

    overall = ok_A and ok_B
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())
