#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T24_SV_commit_rate_record_speed_limit

Single-file, single-run test.

Purpose
-------
Anchor a structural bound: "tick-facts cannot be produced faster than the record
can become distinguishable". This provides a testable scaffold for the claim:

    λ_effective (tick facts) is limited by record dynamics scale ω_record.

CCM link
--------
- Bulk evolves unitarily in τ (bulk-parameter).
- Commit events occur as a Poisson stopping-time process with total rate λ.
- Outcomes come from record physics (Instrument). A "tick" is a record-class that
  must be physically distinguishable, not a label you can call early.

Model
-----
Record is a 2-level system with Hamiltonian:
    H = (ħΩ/2) σ_x
Initial record state: |0>
Unitary evolution:
    |ψ(τ)> = cos(Ωτ/2)|0> - i sin(Ωτ/2)|1>
So:
    w_tick(τ) = |<1|ψ(τ)>|^2 = sin^2(Ωτ/2)
    overlap with initial: |<0|ψ(τ)>|^2 = cos^2(Ωτ/2)

We define "record distinguishability threshold" δ in terms of fidelity with initial:
    |<0|ψ(τ)>|^2 <= δ
This defines a minimal record time:
    τ_min(δ) = 2 * arccos(sqrt(δ)) / Ω

This equals the Mandelstam–Tamm quantum speed limit time for reaching angle
θ = arccos(sqrt(δ)) with energy uncertainty ΔE = ħΩ/2 (saturated for this model).

Test logic
----------
Part A (DI-consistent record): gate tick until τ >= τ_min(δ).
Even if λ is huge, mean tick time should saturate near τ_min(δ) (cannot go below).

Part B (negative control / "unstable record"): no gating; tick can occur at any τ
with probability w_tick(τ). Then for large λ the mean tick time can drop below τ_min(δ).
This demonstrates why "tick" must be a record-class, not an arbitrary projector at any τ.

Deps
----
numpy only.

References (physics background)
-------------------------------
- Mandelstam & Tamm (1945): time-energy uncertainty / speed limit
- Deffner & Campbell (2017): review of quantum speed limits
"""

import math
import numpy as np


def tau_min_from_delta(Omega: float, delta: float) -> float:
    # Require |<0|ψ(τ)>|^2 <= delta
    # cos^2(Omega τ / 2) <= delta  ->  Omega τ / 2 >= arccos(sqrt(delta))
    return 2.0 * math.acos(math.sqrt(delta)) / Omega


def w_tick(Omega: float, tau: float) -> float:
    # sin^2(Omega τ / 2)
    s = math.sin(0.5 * Omega * tau)
    return s * s


def simulate_tick_times(
    rng: np.random.Generator,
    trials: int,
    Omega: float,
    lam: float,
    delta: float,
    gated: bool,
) -> np.ndarray:
    """
    Simulate first 'tick' time in τ.

    Commit events are Poisson with total rate lam:
        inter-arrival times Δ ~ Exp(lam)

    At each Commit event time τ:
      - if gated and τ < τ_min(delta): tick prob = 0
      - else tick prob = w_tick(τ)
    If tick doesn't happen, we continue (the commit outcome is "wait").

    Returns: array of τ_tick for each trial.
    """
    if lam <= 0:
        raise ValueError("lam must be > 0")

    tmin = tau_min_from_delta(Omega, delta) if gated else 0.0
    out = np.empty(trials, dtype=float)

    for k in range(trials):
        tau = 0.0
        while True:
            tau += rng.exponential(scale=1.0 / lam)  # next Commit time
            if tau < tmin:
                continue  # tick forbidden by record distinguishability
            wt = w_tick(Omega, tau)
            if rng.random() < wt:
                out[k] = tau
                break

    return out


def main() -> int:
    seed = 1
    rng = np.random.default_rng(seed)

    # Record dynamics scale (ω_record)
    Omega = 1.0  # choose units where Omega=1 sets τ-scale
    delta = 1e-4  # distinguishability threshold

    trials = 50_000
    lam_list = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]

    tmin = tau_min_from_delta(Omega, delta)

    print("=== T24_SV_commit_rate_record_speed_limit ===")
    print(f"seed={seed}, trials={trials}")
    print(f"Omega={Omega:.6f}  (record frequency scale ω_record)")
    print(f"delta={delta:.1e}  (fidelity threshold |<0|ψ>|^2 <= delta)")
    print(f"tau_min(delta) = {tmin:.6f}")
    print()

    # -----------------------------
    # Part A: gated (DI-consistent record)
    # -----------------------------
    print("PART A: gated tick (record must be distinguishable)")
    print("   lam |  mean(tau_tick) |   std |  mean/tau_min")
    means_A = []
    for lam in lam_list:
        tau_ticks = simulate_tick_times(rng, trials, Omega, lam, delta, gated=True)
        m = float(np.mean(tau_ticks))
        s = float(np.std(tau_ticks, ddof=0))
        means_A.append(m)
        print(f"{lam:6.2f} | {m:14.6f} | {s:6.3f} | {m/tmin:13.6f}")
    print()

    # Check saturation at large λ
    # For very large λ, mean tick time should be close to tau_min(delta).
    lam_hi = lam_list[-1]
    mean_hi = means_A[-1]
    rel_err_hi = abs(mean_hi - tmin) / tmin

    print("CHECK A1: large-lambda saturation")
    print(f"  lam_hi = {lam_hi:.2f}")
    print(f"  mean_hi = {mean_hi:.6f}")
    print(f"  tau_min = {tmin:.6f}")
    print(f"  relative error = {rel_err_hi:.6f}")
    ok_A1 = rel_err_hi < 0.03  # 3% tolerance
    print(f"  RESULT: {'PASS' if ok_A1 else 'FAIL'}  (mean -> tau_min as lam -> large)")
    print()

    # Check that mean never goes below tau_min (up to sampling noise)
    print("CHECK A2: no-superluminal-record (mean not below tau_min)")
    min_mean_A = min(means_A)
    # allow a tiny numerical margin
    ok_A2 = min_mean_A >= 0.999 * tmin
    print(f"  min mean(tau_tick) over lam_list = {min_mean_A:.6f}")
    print(f"  tau_min(delta)                  = {tmin:.6f}")
    print(f"  RESULT: {'PASS' if ok_A2 else 'FAIL'}")
    print()

    # -----------------------------
    # Part B: ungated (negative control)
    # -----------------------------
    print("PART B: ungated tick (negative control: 'unstable record')")
    print("   lam |  mean(tau_tick) |   std |  mean/tau_min")
    means_B = []
    for lam in lam_list:
        tau_ticks = simulate_tick_times(rng, trials, Omega, lam, delta, gated=False)
        m = float(np.mean(tau_ticks))
        s = float(np.std(tau_ticks, ddof=0))
        means_B.append(m)
        print(f"{lam:6.2f} | {m:14.6f} | {s:6.3f} | {m/tmin:13.6f}")
    print()

    print("CHECK B1: negative control shows sub-tau_min ticks at large lam")
    mean_hi_B = means_B[-1]
    ok_B1 = mean_hi_B < 0.8 * tmin  # should be significantly below tau_min
    print(f"  lam_hi = {lam_hi:.2f}")
    print(f"  mean_hi(ungated) = {mean_hi_B:.6f}")
    print(f"  tau_min(delta)   = {tmin:.6f}")
    print(f"  RESULT: {'PASS' if ok_B1 else 'FAIL'}  (shows why gating matters)")
    print()

    overall = ok_A1 and ok_A2 and ok_B1
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())