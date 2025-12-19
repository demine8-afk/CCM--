#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V02_SV_amplitude_damping

Single-file, single-run test.

Claim
-----
Amplitude damping models spontaneous emission: |1⟩ → |0⟩ with probability γ.

Test logic
----------
Part A: Kraus completeness
Part B: ρ_11 → (1-γ)ρ_11
Part C: ρ_00 → ρ_00 + γρ_11
Part D: Full relaxation γ=1 → ground state
Part E: Coherence decay as √(1-γ)
"""

import numpy as np


def amplitude_damping(rho, gamma):
    """Apply amplitude damping channel."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def main() -> int:
    print("=== V02_SV_amplitude_damping ===")
    print()
    
    # ========================================
    # Part A: Kraus completeness
    # ========================================
    print("-" * 60)
    print("PART A: Kraus operators completeness")
    print("-" * 60)
    
    gamma = 0.4
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    
    completeness = K0.conj().T @ K0 + K1.conj().T @ K1
    err_A = np.linalg.norm(completeness - np.eye(2))
    ok_A = err_A < 1e-10
    
    print(f"  γ = {gamma}")
    print(f"  ||Σ K†K - I|| = {err_A:.2e}")
    print(f"  RESULT: {'PASS' if ok_A else 'FAIL'}")
    print()
    
    # ========================================
    # Part B: ρ_11 decay
    # ========================================
    print("-" * 60)
    print("PART B: ρ_11 → (1-γ)ρ_11")
    print("-" * 60)
    
    rho = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=complex)
    
    print(f"  {'γ':>6} | {'ρ_11 out':>10} | {'(1-γ)ρ_11':>10} | {'error':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    max_err_B = 0.0
    for gamma in [0.0, 0.2, 0.5, 0.8, 1.0]:
        rho_out = amplitude_damping(rho, gamma)
        theory = (1 - gamma) * rho[1, 1]
        err = abs(rho_out[1, 1] - theory)
        max_err_B = max(max_err_B, err)
        print(f"  {gamma:>6.2f} | {rho_out[1,1].real:>10.6f} | {theory.real:>10.6f} | {err:>10.2e}")
    
    ok_B = max_err_B < 1e-10
    print(f"  RESULT: {'PASS' if ok_B else 'FAIL'}")
    print()
    
    # ========================================
    # Part C: ρ_00 gain
    # ========================================
    print("-" * 60)
    print("PART C: ρ_00 → ρ_00 + γρ_11")
    print("-" * 60)
    
    print(f"  {'γ':>6} | {'ρ_00 out':>10} | {'ρ_00+γρ_11':>10} | {'error':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    max_err_C = 0.0
    for gamma in [0.0, 0.2, 0.5, 0.8, 1.0]:
        rho_out = amplitude_damping(rho, gamma)
        theory = rho[0, 0] + gamma * rho[1, 1]
        err = abs(rho_out[0, 0] - theory)
        max_err_C = max(max_err_C, err)
        print(f"  {gamma:>6.2f} | {rho_out[0,0].real:>10.6f} | {theory.real:>10.6f} | {err:>10.2e}")
    
    ok_C = max_err_C < 1e-10
    print(f"  RESULT: {'PASS' if ok_C else 'FAIL'}")
    print()
    
    # ========================================
    # Part D: Full relaxation
    # ========================================
    print("-" * 60)
    print("PART D: γ=1 → ground state")
    print("-" * 60)
    
    rho_excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    rho_out = amplitude_damping(rho_excited, 1.0)
    rho_ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    
    err_D = np.linalg.norm(rho_out - rho_ground)
    ok_D = err_D < 1e-10
    
    print(f"  |1⟩⟨1| after γ=1:")
    print(f"  {rho_out}")
    print(f"  ||ρ_out - |0⟩⟨0||| = {err_D:.2e}")
    print(f"  RESULT: {'PASS' if ok_D else 'FAIL'}")
    print()
    
    # ========================================
    # Part E: Coherence decay
    # ========================================
    print("-" * 60)
    print("PART E: |ρ_01| → √(1-γ)|ρ_01|")
    print("-" * 60)
    
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rho_01_orig = abs(rho[0, 1])
    
    print(f"  {'γ':>6} | {'|ρ_01| out':>10} | {'√(1-γ)|ρ_01|':>12} | {'error':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
    
    max_err_E = 0.0
    for gamma in [0.0, 0.2, 0.5, 0.8, 0.99]:
        rho_out = amplitude_damping(rho, gamma)
        theory = np.sqrt(1 - gamma) * rho_01_orig
        err = abs(abs(rho_out[0, 1]) - theory)
        max_err_E = max(max_err_E, err)
        print(f"  {gamma:>6.2f} | {abs(rho_out[0,1]):>10.6f} | {theory:>12.6f} | {err:>10.2e}")
    
    ok_E = max_err_E < 1e-10
    print(f"  RESULT: {'PASS' if ok_E else 'FAIL'}")
    print()
    
    # ========================================
    # VERDICT
    # ========================================
    print("=" * 60)
    overall = ok_A and ok_B and ok_C and ok_D and ok_E
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())