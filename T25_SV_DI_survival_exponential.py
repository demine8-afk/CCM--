#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T25_SV_DI_survival_exponential

Single-file, single-run test.

Claim
-----
DI-segmentation uniquely fixes exponential survival:
  S(Δτ₁ + Δτ₂) = S(Δτ₁) * S(Δτ₂)
with S(0)=1, S continuous, 0 < S ≤ 1 implies S(Δτ) = exp(-λΔτ).

Test logic
----------
Part A: Verify that exponential form satisfies the functional equation exactly.

Part B: Verify that power-law S(Δτ) = (1 + αΔτ)^{-β} violates the equation
        (negative control).

Part C: Numerical stability - segmentation into N pieces should give same result.
"""

import numpy as np

def main() -> int:
    print("=== T25_SV_DI_survival_exponential ===")
    print()
    
    seed = 42
    rng = np.random.default_rng(seed)
    trials = 1000
    
    # -----------------------------
    # Part A: Exponential satisfies S(x+y) = S(x)*S(y)
    # -----------------------------
    print("PART A: Exponential form S(Δτ) = exp(-λΔτ)")
    
    lam = 1.0
    max_err_A = 0.0
    
    for _ in range(trials):
        x = rng.uniform(0, 10)
        y = rng.uniform(0, 10)
        
        S_sum = np.exp(-lam * (x + y))
        S_prod = np.exp(-lam * x) * np.exp(-lam * y)
        
        err = abs(S_sum - S_prod)
        max_err_A = max(max_err_A, err)
    
    print(f"  λ = {lam}")
    print(f"  trials = {trials}")
    print(f"  max|S(x+y) - S(x)*S(y)| = {max_err_A:.3e}")
    ok_A = max_err_A < 1e-14
    print(f"  RESULT: {'PASS' if ok_A else 'FAIL'}")
    print()
    
    # -----------------------------
    # Part B: Power-law violates (negative control)
    # -----------------------------
    print("PART B: Power-law S(Δτ) = (1 + αΔτ)^{-β} (negative control)")
    
    alpha = 0.5
    beta = 2.0
    
    def S_power(t):
        return (1 + alpha * t) ** (-beta)
    
    max_violation = 0.0
    
    for _ in range(trials):
        x = rng.uniform(0.1, 5)  # avoid x=0 where trivially satisfied
        y = rng.uniform(0.1, 5)
        
        S_sum = S_power(x + y)
        S_prod = S_power(x) * S_power(y)
        
        violation = abs(S_sum - S_prod)
        max_violation = max(max_violation, violation)
    
    print(f"  α = {alpha}, β = {beta}")
    print(f"  max|S(x+y) - S(x)*S(y)| = {max_violation:.3e}")
    ok_B = max_violation > 0.01  # must show significant violation
    print(f"  RESULT: {'PASS' if ok_B else 'FAIL'} (violation detected)")
    print()
    
    # -----------------------------
    # Part C: Segmentation invariance for exponential
    # -----------------------------
    print("PART C: Segmentation invariance (exponential)")
    
    Delta_tau = 5.0
    lam = 1.0
    S_direct = np.exp(-lam * Delta_tau)
    
    print(f"  Δτ = {Delta_tau}, λ = {lam}")
    print(f"  S_direct = exp(-λΔτ) = {S_direct:.10f}")
    print()
    print("  N segments | S_product      | rel_error")
    
    max_rel_err = 0.0
    for N in [2, 5, 10, 100, 1000, 10000]:
        dt = Delta_tau / N
        S_segment = np.exp(-lam * dt)
        S_product = S_segment ** N
        
        rel_err = abs(S_product - S_direct) / S_direct
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  {N:10d} | {S_product:.10f} | {rel_err:.3e}")
    
    ok_C = max_rel_err < 1e-10
    print(f"  max rel_error = {max_rel_err:.3e}")
    print(f"  RESULT: {'PASS' if ok_C else 'FAIL'}")
    print()
    
    # -----------------------------
    # Part D: Power-law fails segmentation (negative control)
    # -----------------------------
    print("PART D: Power-law fails segmentation (negative control)")
    
    S_direct_power = S_power(Delta_tau)
    print(f"  S_power({Delta_tau}) = {S_direct_power:.10f}")
    print()
    print("  N segments | S_product      | rel_error")
    
    for N in [2, 5, 10, 100]:
        dt = Delta_tau / N
        S_segment = S_power(dt)
        S_product = S_segment ** N
        
        rel_err = abs(S_product - S_direct_power) / S_direct_power
        print(f"  {N:10d} | {S_product:.10f} | {rel_err:.3e}")
    
    # For power-law, error should grow with N
    dt_100 = Delta_tau / 100
    S_prod_100 = S_power(dt_100) ** 100
    rel_err_100 = abs(S_prod_100 - S_direct_power) / S_direct_power
    
    ok_D = rel_err_100 > 0.1  # substantial error
    print(f"  RESULT: {'PASS' if ok_D else 'FAIL'} (segmentation-dependence detected)")
    print()
    
    # -----------------------------
    # Overall
    # -----------------------------
    overall = ok_A and ok_B and ok_C and ok_D
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())