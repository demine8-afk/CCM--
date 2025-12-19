#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T22_SV_commit_hazard_DI_refinement

Single-file, single-run test.

Purpose
-------
Show Description Invariance (DI) for a hazard-based commit model under refinement.

We consider a single physical interval Δt with constant hazard λ.
A "fact" is: did commit happen within Δt?

Correct DI-safe refinement:
  p_step(N) = 1 - exp(-λ Δt / N)
  P(commit within Δt) = 1 - (1 - p_step(N))^N = 1 - exp(-λ Δt)  (independent of N)

Negative control (intentionally WRONG refinement):
  keep p_step fixed as the coarse probability p_coarse = 1 - exp(-λ Δt)
  then P_total(N) = 1 - (1 - p_coarse)^N depends on N -> DI violation

No external deps beyond numpy.
"""

import numpy as np
import math

def estimate_commit_within_dt(rng, lam, dt, N, samples, mode):
    """
    mode:
      'correct'  : p_step = 1 - exp(-lam*dt/N)
      'wrong'    : p_step = 1 - exp(-lam*dt)   (kept fixed per segment; DI-breaking)
    """
    if mode == 'correct':
        p_step = 1.0 - math.exp(-lam * dt / N)
    elif mode == 'wrong':
        p_step = 1.0 - math.exp(-lam * dt)
    else:
        raise ValueError("mode must be 'correct' or 'wrong'")

    # Bernoulli trials across N segments: event occurs if any segment triggers
    # (This is the discrete-time implementation of a stopping-time within dt.)
    u = rng.random((samples, N))
    occurred = (u < p_step).any(axis=1)
    return float(occurred.mean())

def main():
    seed = 1
    samples = 300_000
    N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Choose lam*dt not tiny so wrong refinement is visibly non-invariant.
    lam = 0.8
    dt  = 1.0
    analytic = 1.0 - math.exp(-lam * dt)

    rng = np.random.default_rng(seed)

    print("=== T22_SV_commit_hazard_DI_refinement ===")
    print(f"seed={seed}, samples={samples}")
    print(f"lambda={lam:.4f}, dt={dt:.4f}, analytic P(commit within dt) = {analytic:.6f}")
    print(f"N_LIST = {N_LIST}")
    print()

    # ---------------- Correct refinement ----------------
    print("MODEL A: DI-safe hazard refinement (correct)")
    print("    N |  P_hat(commit<=dt) | |P_hat - analytic|")
    diffs = []
    for N in N_LIST:
        p_hat = estimate_commit_within_dt(rng, lam, dt, N, samples, mode='correct')
        diff = abs(p_hat - analytic)
        diffs.append(diff)
        print(f"{N:5d} | {p_hat:16.6f} | {diff:16.6e}")
    max_diff = max(diffs)
    ok = max_diff < 5e-3
    print(f"max|P_hat - analytic| = {max_diff:.3e}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print()

    # ---------------- Negative control ----------------
    print("NEGATIVE CONTROL: WRONG refinement (p_step fixed, DI-breaking)")
    print("    N |  P_hat(commit<=dt)")
    p_vals = []
    for N in N_LIST:
        p_hat = estimate_commit_within_dt(rng, lam, dt, N, samples, mode='wrong')
        p_vals.append(p_hat)
        print(f"{N:5d} | {p_hat:16.6f}")

    # Check that it actually depends on N (difference between N=1 and N=max)
    dep = abs(p_vals[0] - p_vals[-1])
    print(f"|P_hat(N=1) - P_hat(N={N_LIST[-1]})| = {dep:.3e}")
    neg_ok = dep > 1e-2
    print(f"NEGATIVE CONTROL: {'PASS' if neg_ok else 'FAIL'} (expected FAIL observed)")
    print(f"OVERALL: {'PASS' if (ok and neg_ok) else 'FAIL'}")
    return 0 if (ok and neg_ok) else 1

if __name__ == "__main__":
    raise SystemExit(main())
