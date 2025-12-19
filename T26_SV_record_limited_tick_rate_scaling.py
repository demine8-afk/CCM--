#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T26_SV_record_limited_tick_rate_scaling

Single-file, single-run test.

Purpose
-------
Anchor the structural statement:
  Even if total Commit hazard λ is very large, the *tick* Fact rate cannot exceed
  the record dynamics scale ω_record.

In a minimal 2-level record model:
  H_R = (ħΩ/2) σ_x, |r0>=|0>
  Fidelity with initial: F(τ)=cos^2(Ωτ/2)

Define distinguishability threshold δ:
  tick can be physical only after F(τ) <= δ
  τ_min(δ) = 2 arccos(sqrt(δ)) / Ω

Then for large λ, mean tick interval saturates:
  <τ_tick> -> τ_min(δ)

So max tick rate scales:
  r_tick_max ~ 1/τ_min ∝ Ω

We test:
(A) scaling: <τ_tick> ∝ 1/Ω at large λ (slope ~ -1 in log-log)
(B) saturation: increasing λ above O(Ω) barely changes <τ_tick>
(C) negative control (ungated): for large λ, <τ_tick> collapses ~ O(1/λ) -> unphysical

Deps: numpy only.
"""

import math
import numpy as np


def fit_slope_loglog(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = np.log(x)
    ly = np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(intercept)


def tau_min_from_delta(Omega: float, delta: float) -> float:
    return 2.0 * math.acos(math.sqrt(delta)) / Omega


def w_tick(Omega: float, tau: float) -> float:
    s = math.sin(0.5 * Omega * tau)
    return s * s


def simulate_one_tick_time(rng, Omega: float, lam: float, delta: float, gated: bool) -> float:
    """Simulate first tick time in τ."""
    tmin = tau_min_from_delta(Omega, delta) if gated else 0.0
    tau = 0.0
    while True:
        tau += rng.exponential(scale=1.0 / lam)
        if tau < tmin:
            continue
        if rng.random() < w_tick(Omega, tau):
            return tau


def simulate_tick_intervals(rng, trials: int, Omega: float, lam: float, delta: float, gated: bool) -> np.ndarray:
    """Each interval starts from fresh record |0> (post-Commit reset)."""
    out = np.empty(trials, dtype=float)
    for k in range(trials):
        out[k] = simulate_one_tick_time(rng, Omega, lam, delta, gated)
    return out


def main() -> int:
    seed = 1
    rng = np.random.default_rng(seed)

    delta = 1e-4
    trials = 50_000

    Omega_list = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    print("=== T26_SV_record_limited_tick_rate_scaling ===")
    print(f"seed={seed}, trials={trials}, delta={delta:.1e}")
    print()

    # ------------------------------------------------------------
    # Part A: Scaling with Omega at large lambda
    # ------------------------------------------------------------
    print("PART A: scaling <tau_tick> ~ const/Omega at large lambda")
    print(" Omega |      lam |  tau_min | mean_tau | mean/tau_min")
    means = []
    tmins = []
    for Omega in Omega_list:
        # choose very large lambda relative to Omega to force saturation
        lam = 200.0 * Omega
        tmin = tau_min_from_delta(Omega, delta)
        taus = simulate_tick_intervals(rng, trials, Omega, lam, delta, gated=True)
        mean_tau = float(np.mean(taus))
        means.append(mean_tau)
        tmins.append(tmin)
        print(f"{Omega:5.2f} | {lam:8.2f} | {tmin:8.4f} | {mean_tau:8.4f} | {mean_tau/tmin:12.6f}")

    slope, intercept = fit_slope_loglog(Omega_list, means)
    print()
    print("CHECK A1: log-log slope of mean_tau vs Omega")
    print(f"  slope = {slope:+.4f}  (expected ~ -1.0)")
    ok_A1 = abs(slope + 1.0) < 0.05
    print(f"  RESULT: {'PASS' if ok_A1 else 'FAIL'}")
    print()

    # also check closeness to tau_min across Omegas
    rel_errs = [abs(m - t)/t for m, t in zip(means, tmins)]
    max_rel = max(rel_errs)
    print("CHECK A2: saturation mean_tau -> tau_min across Omega")
    print(f"  max relative error = {max_rel:.6f} (expected small)")
    ok_A2 = max_rel < 0.01
    print(f"  RESULT: {'PASS' if ok_A2 else 'FAIL'}")
    print()

    # ------------------------------------------------------------
    # Part B: Saturation with lambda for fixed Omega
    # ------------------------------------------------------------
    Omega0 = 1.0
    tmin0 = tau_min_from_delta(Omega0, delta)
    lam_factors = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]

    print("PART B: saturation in lambda (fixed Omega=1)")
    print(" lam/Omega |    lam |  tau_min | mean_tau | mean/tau_min")
    means_B = []
    for f in lam_factors:
        lam = f * Omega0
        taus = simulate_tick_intervals(rng, trials, Omega0, lam, delta, gated=True)
        mean_tau = float(np.mean(taus))
        means_B.append(mean_tau)
        print(f"{f:9.2f} | {lam:6.2f} | {tmin0:8.4f} | {mean_tau:8.4f} | {mean_tau/tmin0:12.6f}")

    # compare last two as "already saturated"
    m50 = means_B[lam_factors.index(50.0)]
    m200 = means_B[lam_factors.index(200.0)]
    rel_sat = abs(m50 - m200) / m200

    print()
    print("CHECK B1: increasing lambda beyond ~50*Omega barely changes mean")
    print(f"  mean(50Ω)  = {m50:.6f}")
    print(f"  mean(200Ω) = {m200:.6f}")
    print(f"  relative diff = {rel_sat:.6f}")
    ok_B1 = rel_sat < 0.01
    print(f"  RESULT: {'PASS' if ok_B1 else 'FAIL'}")
    print()

    # ------------------------------------------------------------
    # Part C: Negative control (ungated)
    # ------------------------------------------------------------
    print("PART C: negative control (ungated) shows unphysical collapse ~ 1/lambda")
    print(" lam/Omega |    lam | mean_tau_ungated")
    ung = []
    for f in [10.0, 50.0, 200.0]:
        lam = f * Omega0
        taus = simulate_tick_intervals(rng, trials, Omega0, lam, delta, gated=False)
        mean_tau = float(np.mean(taus))
        ung.append(mean_tau)
        print(f"{f:9.2f} | {lam:6.2f} | {mean_tau:14.6f}")

    # expect ungated mean at 200Ω much smaller than tau_min
    ok_C1 = ung[-1] < 0.3 * tmin0
    print()
    print("CHECK C1: ungated mean << tau_min at large lambda (should be TRUE)")
    print(f"  mean_ungated(200Ω) = {ung[-1]:.6f}")
    print(f"  tau_min            = {tmin0:.6f}")
    print(f"  RESULT: {'PASS' if ok_C1 else 'FAIL'}")
    print()

    overall = ok_A1 and ok_A2 and ok_B1 and ok_C1
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())