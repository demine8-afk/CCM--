#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V01_SV_dephasing_channel

Single-file, single-run test.

Claim
-----
Dephasing channel suppresses off-diagonal elements while preserving diagonal.

Test logic
----------
Part A: Kraus operators satisfy completeness
Part B: Diagonal elements preserved
Part C: Off-diagonal elements suppressed by (1-p)
Part D: Negative control - wrong Kraus breaks CPTP
"""

import numpy as np


def dephasing_channel(rho, p):
    """Apply dephasing channel with parameter p."""
    K0 = np.sqrt(1 - p) * np.eye(2)
    K1 = np.sqrt(p) * np.array([[1, 0], [0, 0]])
    K2 = np.sqrt(p) * np.array([[0, 0], [0, 1]])
    
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T + K2 @ rho @ K2.conj().T


def main() -> int:
    print("=== V01_SV_dephasing_channel ===")
    print()
    
    # ========================================
    # Part A: Kraus completeness
    # ========================================
    print("-" * 60)
    print("PART A: Kraus operators completeness")
    print("-" * 60)
    
    p = 0.3
    K0 = np.sqrt(1 - p) * np.eye(2)
    K1 = np.sqrt(p) * np.array([[1, 0], [0, 0]])
    K2 = np.sqrt(p) * np.array([[0, 0], [0, 1]])
    
    completeness = K0.conj().T @ K0 + K1.conj().T @ K1 + K2.conj().T @ K2
    err_A = np.linalg.norm(completeness - np.eye(2))
    ok_A = err_A < 1e-10
    
    print(f"  Σ K†K =")
    print(f"  {completeness}")
    print(f"  ||Σ K†K - I|| = {err_A:.2e}")
    print(f"  RESULT: {'PASS' if ok_A else 'FAIL'}")
    print()
    
    # ========================================
    # Part B: Diagonal preserved
    # ========================================
    print("-" * 60)
    print("PART B: Diagonal elements preserved")
    print("-" * 60)
    
    rho = np.array([[0.7, 0.4], [0.4, 0.3]], dtype=complex)
    
    max_err_diag = 0.0
    for p in [0.0, 0.3, 0.5, 0.9, 1.0]:
        rho_out = dephasing_channel(rho, p)
        err_00 = abs(rho_out[0, 0] - rho[0, 0])
        err_11 = abs(rho_out[1, 1] - rho[1, 1])
        max_err_diag = max(max_err_diag, err_00, err_11)
    
    ok_B = max_err_diag < 1e-10
    
    print(f"  ρ_00 = {rho[0,0]:.4f}, ρ_11 = {rho[1,1]:.4f}")
    print(f"  max |Δρ_diag| over p = {max_err_diag:.2e}")
    print(f"  RESULT: {'PASS' if ok_B else 'FAIL'}")
    print()
    
    # ========================================
    # Part C: Off-diagonal suppressed
    # ========================================
    print("-" * 60)
    print("PART C: Off-diagonal suppressed by (1-p)")
    print("-" * 60)
    
    print(f"  {'p':>6} | {'|ρ_01|':>10} | {'(1-p)|ρ_01|':>12} | {'error':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
    
    rho_01_orig = abs(rho[0, 1])
    max_err_off = 0.0
    
    for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
        rho_out = dephasing_channel(rho, p)
        rho_01_out = abs(rho_out[0, 1])
        theory = (1 - p) * rho_01_orig
        err = abs(rho_01_out - theory)
        max_err_off = max(max_err_off, err)
        print(f"  {p:>6.2f} | {rho_01_out:>10.6f} | {theory:>12.6f} | {err:>10.2e}")
    
    ok_C = max_err_off < 1e-10
    
    print(f"  RESULT: {'PASS' if ok_C else 'FAIL'}")
    print()
    
    # ========================================
    # Part D: Negative control
    # ========================================
    print("-" * 60)
    print("PART D: Negative control (wrong Kraus)")
    print("-" * 60)
    
    K0_bad = np.sqrt(1.5) * np.eye(2)
    K1_bad = np.sqrt(0.5) * np.array([[1, 0], [0, -1]])
    
    completeness_bad = K0_bad.conj().T @ K0_bad + K1_bad.conj().T @ K1_bad
    err_bad = np.linalg.norm(completeness_bad - np.eye(2))
    
    ok_D = err_bad > 0.1
    
    print(f"  Bad Kraus: ||Σ K†K - I|| = {err_bad:.4f}")
    print(f"  RESULT: {'PASS' if ok_D else 'FAIL'} (violation detected)")
    print()
    
    # ========================================
    # VERDICT
    # ========================================
    print("=" * 60)
    overall = ok_A and ok_B and ok_C and ok_D
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())