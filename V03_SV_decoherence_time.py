#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V03_SV_decoherence_time

Single-file, single-run test.

Claim
-----
Coherence decays exponentially with time constant T_2 = 1/Γ.

Test logic
----------
Part A: Coherence decay is exponential
Part B: T_2 extracted from fit matches 1/Γ
Part C: Different Γ values give correct T_2
"""

import numpy as np


def evolve_dephasing(rho, Gamma, t, dt=0.01):
    """Evolve under continuous dephasing for time t."""
    n_steps = int(t / dt)
    p_per_step = 1 - np.exp(-Gamma * dt)
    
    rho = rho.copy()
    for _ in range(n_steps):
        rho[0, 1] *= (1 - p_per_step)
        rho[1, 0] *= (1 - p_per_step)
    
    return rho


def main() -> int:
    print("=== V03_SV_decoherence_time ===")
    print()
    
    # ========================================
    # Part A: Exponential decay
    # ========================================
    print("-" * 60)
    print("PART A: Coherence decay is exponential")
    print("-" * 60)
    
    Gamma = 1.0
    T_2_theory = 1.0 / Gamma
    
    times = np.linspace(0, 5 * T_2_theory, 50)
    coherences = []
    
    for t in times:
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        rho = evolve_dephasing(rho, Gamma, t)
        coherences.append(abs(rho[0, 1]))
    
    coherences = np.array(coherences)
    theory = 0.5 * np.exp(-Gamma * times)
    
    max_err = np.max(np.abs(coherences - theory))
    ok_A = max_err < 0.01
    
    print(f"  Γ = {Gamma}, T_2 = {T_2_theory}")
    print(f"  max |coherence - 0.5*exp(-Γt)| = {max_err:.4f}")
    print(f"  RESULT: {'PASS' if ok_A else 'FAIL'}")
    print()
    
    # ========================================
    # Part B: T_2 from fit
    # ========================================
    print("-" * 60)
    print("PART B: T_2 extracted from fit")
    print("-" * 60)
    
    log_coh = np.log(coherences + 1e-10)
    coeffs = np.polyfit(times, log_coh, 1)
    T_2_fit = -1.0 / coeffs[0]
    
    err_T2 = abs(T_2_fit - T_2_theory) / T_2_theory
    ok_B = err_T2 < 0.01
    
    print(f"  T_2 (theory) = {T_2_theory:.4f}")
    print(f"  T_2 (fit)    = {T_2_fit:.4f}")
    print(f"  Relative error = {err_T2:.4f}")
    print(f"  RESULT: {'PASS' if ok_B else 'FAIL'}")
    print()
    
    # ========================================
    # Part C: Different Γ values
    # ========================================
    print("-" * 60)
    print("PART C: T_2 = 1/Γ for various Γ")
    print("-" * 60)
    
    print(f"  {'Γ':>8} | {'T_2 theory':>10} | {'T_2 fit':>10} | {'rel_err':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    max_err_C = 0.0
    for Gamma in [0.5, 1.0, 2.0, 5.0]:
        T_2_th = 1.0 / Gamma
        times = np.linspace(0, 5 * T_2_th, 50)
        coherences = []
        
        for t in times:
            rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
            rho = evolve_dephasing(rho, Gamma, t)
            coherences.append(abs(rho[0, 1]))
        
        log_coh = np.log(np.array(coherences) + 1e-10)
        coeffs = np.polyfit(times, log_coh, 1)
        T_2_f = -1.0 / coeffs[0]
        
        rel_err = abs(T_2_f - T_2_th) / T_2_th
        max_err_C = max(max_err_C, rel_err)
        
        print(f"  {Gamma:>8.2f} | {T_2_th:>10.4f} | {T_2_f:>10.4f} | {rel_err:>10.4f}")
    
    ok_C = max_err_C < 0.02
    print(f"  RESULT: {'PASS' if ok_C else 'FAIL'}")
    print()
    
    # ========================================
    # VERDICT
    # ========================================
    print("=" * 60)
    overall = ok_A and ok_B and ok_C
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())