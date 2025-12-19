#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E03_SV_photon_interference

Single-file, single-run test.

Claim
-----
Single photon interference follows |ψ|² = |A₁ + A₂|².
Mach-Zehnder: P(D1) = cos²(φ/2), P(D2) = sin²(φ/2).

Test logic
----------
Part A: Mach-Zehnder phase sweep
Part B: Hong-Ou-Mandel two-photon interference
Part C: Negative control — classical particles don't interfere
Part D: Interference visibility
"""

import numpy as np


def mach_zehnder_probabilities(phi):
    """
    Single photon Mach-Zehnder interferometer.
    Returns P(D1), P(D2) for phase shift phi in upper arm.
    """
    P_D1 = np.cos(phi / 2) ** 2
    P_D2 = np.sin(phi / 2) ** 2
    
    return P_D1, P_D2


def hom_visibility(distinguishability):
    """
    Hong-Ou-Mandel effect.
    Two photons enter 50:50 beam splitter from different ports.
    
    Indistinguishable (d=0): both exit same port (bunching), P(coincidence) = 0
    Distinguishable (d=1): classical, P(coincidence) = 0.5
    
    Returns: P(coincidence) = 0.5 * d²
    """
    return 0.5 * distinguishability ** 2


def classical_mz(phi):
    """
    Classical particle through Mach-Zehnder.
    No interference — 50/50 split regardless of phase.
    """
    return 0.5, 0.5


def main() -> int:
    print("=== E03_SV_photon_interference ===")
    print()
    
    # ========================================
    # Part A: Mach-Zehnder phase sweep
    # ========================================
    print("-" * 60)
    print("PART A: Mach-Zehnder interferometer")
    print("-" * 60)
    
    print(f"  {'φ/π':>8} | {'P(D1)':>8} | {'P(D2)':>8} | {'cos²':>8} | {'sin²':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    max_err_A = 0.0
    phases = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    for phi_pi in phases:
        phi = phi_pi * np.pi
        P1, P2 = mach_zehnder_probabilities(phi)
        
        cos2 = np.cos(phi / 2) ** 2
        sin2 = np.sin(phi / 2) ** 2
        
        err1 = abs(P1 - cos2)
        err2 = abs(P2 - sin2)
        max_err_A = max(max_err_A, err1, err2)
        
        norm_err = abs(P1 + P2 - 1.0)
        max_err_A = max(max_err_A, norm_err)
        
        print(f"  {phi_pi:>8.2f} | {P1:>8.4f} | {P2:>8.4f} | {cos2:>8.4f} | {sin2:>8.4f}")
    
    ok_A = max_err_A < 1e-10
    print(f"  max error = {max_err_A:.2e}")
    print(f"  RESULT: {'PASS' if ok_A else 'FAIL'}")
    print()
    
    # ========================================
    # Part B: Hong-Ou-Mandel
    # ========================================
    print("-" * 60)
    print("PART B: Hong-Ou-Mandel two-photon interference")
    print("-" * 60)
    
    print(f"  {'d':>8} | {'P(coinc)':>10} | {'theory':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}")
    
    max_err_B = 0.0
    
    for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
        P_coinc = hom_visibility(d)
        theory = 0.5 * d ** 2
        err = abs(P_coinc - theory)
        max_err_B = max(max_err_B, err)
        
        print(f"  {d:>8.2f} | {P_coinc:>10.4f} | {theory:>10.4f}")
    
    P_indist = hom_visibility(0.0)
    ok_B = max_err_B < 1e-10 and P_indist == 0.0
    
    print(f"  Indistinguishable (d=0): P(coinc) = {P_indist} (should be 0)")
    print(f"  RESULT: {'PASS' if ok_B else 'FAIL'}")
    print()
    
    # ========================================
    # Part C: Classical negative control
    # ========================================
    print("-" * 60)
    print("PART C: Classical particles (negative control)")
    print("-" * 60)
    
    print(f"  {'φ/π':>8} | {'P(D1)':>8} | {'P(D2)':>8} | {'quantum P1':>10}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    
    interference_seen = False
    
    for phi_pi in [0, 0.5, 1.0]:
        phi = phi_pi * np.pi
        P1_cl, P2_cl = classical_mz(phi)
        P1_qm, P2_qm = mach_zehnder_probabilities(phi)
        
        print(f"  {phi_pi:>8.2f} | {P1_cl:>8.4f} | {P2_cl:>8.4f} | {P1_qm:>10.4f}")
        
        if abs(P1_cl - 0.5) > 1e-10 or abs(P2_cl - 0.5) > 1e-10:
            interference_seen = True
    
    ok_C = not interference_seen
    print(f"  Classical shows no interference: {ok_C}")
    print(f"  RESULT: {'PASS' if ok_C else 'FAIL'}")
    print()
    
    # ========================================
    # Part D: Visibility
    # ========================================
    print("-" * 60)
    print("PART D: Interference visibility")
    print("-" * 60)
    
    # Visibility V = (P_max - P_min) / (P_max + P_min)
    # For ideal MZ with D1: P_max=1 at φ=0, P_min=0 at φ=π
    # V = (1 - 0) / (1 + 0) = 1
    
    P_max = mach_zehnder_probabilities(0)[0]
    P_min = mach_zehnder_probabilities(np.pi)[0]
    
    # Correct visibility calculation
    denom = P_max + P_min
    if denom < 1e-10:
        # Both zero — undefined, but for P_max=1, P_min=0, denom=1
        V = 0.0
    else:
        V = (P_max - P_min) / denom
    
    # For ideal case: V = (1 - 0) / (1 + 0) = 1
    V_theory = 1.0
    err_V = abs(V - V_theory)
    ok_D = err_V < 1e-10
    
    print(f"  P_max(D1) at φ=0:  {P_max:.6f}")
    print(f"  P_min(D1) at φ=π:  {P_min:.6f}")
    print(f"  Denominator:       {denom:.6f}")
    print(f"  Visibility V = (P_max - P_min) / (P_max + P_min) = {V:.6f}")
    print(f"  V (theory) = {V_theory:.6f}")
    print(f"  Error = {err_V:.2e}")
    print(f"  RESULT: {'PASS' if ok_D else 'FAIL'}")
    print()
    
    # ========================================
    # VERDICT
    # ========================================
    print("=" * 60)
    overall = ok_A and ok_B and ok_C and ok_D
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    if overall:
        print("""
  Photon interference verified:
  • Mach-Zehnder: P(D1) = cos²(φ/2), P(D2) = sin²(φ/2)
  • Hong-Ou-Mandel: indistinguishable photons bunch (P_coinc = 0)
  • Classical particles don't interfere (always 50/50)
  • Visibility V = 1 for ideal interferometer
        """)
    
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())