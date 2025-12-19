#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T29_SV_bell_chsh.py

Single-file, single-run test.

Purpose
-------
Verify CCM reproduces Bell/CHSH predictions:
1. Singlet gives |S| = 2√2 (Tsirelson bound, violates classical |S| ≤ 2)
2. No-signalling: Alice's marginals independent of Bob's setting
3. Product state satisfies classical bound
4. Werner states: threshold p > 1/√2 for violation

CCM interpretation:
- Singlet is Bulk state (entangled, no Facts yet)
- Alice and Bob have local Instruments (spacelike Commits)
- Correlations from Bulk structure, not signalling
- DI (D4): spacelike Commit order doesn't affect statistics

No external deps beyond numpy.
"""

import numpy as np

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def singlet():
    """Singlet state |ψ⁻⟩ = (|01⟩ - |10⟩)/√2, returns density matrix."""
    psi = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())

def product_00():
    """Product state |00⟩, returns density matrix."""
    psi = np.array([1, 0, 0, 0], dtype=complex)
    return np.outer(psi, psi.conj())

def werner(p):
    """Werner state: p * singlet + (1-p) * I/4"""
    return p * singlet() + (1 - p) * np.eye(4) / 4

def spin_observable(theta):
    """
    Spin observable along direction in xz-plane.
    σ(θ) = cos(θ) Z + sin(θ) X
    Eigenvalues ±1.
    """
    return np.cos(theta) * Z + np.sin(theta) * X

def correlator(rho, theta_A, theta_B):
    """
    E(θ_A, θ_B) = ⟨σ_A(θ_A) ⊗ σ_B(θ_B)⟩
    
    For singlet: E = -cos(θ_A - θ_B)
    """
    A = spin_observable(theta_A)
    B = spin_observable(theta_B)
    AB = np.kron(A, B)
    return np.real(np.trace(AB @ rho))

def chsh(rho, a1, a2, b1, b2):
    """
    CHSH operator expectation:
    S = E(a1,b1) + E(a1,b2) + E(a2,b1) - E(a2,b2)
    
    Classical bound: |S| ≤ 2
    Quantum max (Tsirelson): |S| ≤ 2√2
    """
    E11 = correlator(rho, a1, b1)
    E12 = correlator(rho, a1, b2)
    E21 = correlator(rho, a2, b1)
    E22 = correlator(rho, a2, b2)
    return E11 + E12 + E21 - E22

def marginal_alice(rho, theta_A):
    """
    P(a=±1 | θ_A) from reduced state.
    Returns (P(+1), P(-1)).
    """
    # Reduced state of Alice
    rho_A = np.array([[rho[0,0] + rho[1,1], rho[0,2] + rho[1,3]],
                      [rho[2,0] + rho[3,1], rho[2,2] + rho[3,3]]])
    
    # Projectors for ±1 outcomes
    A = spin_observable(theta_A)
    P_plus = (I2 + A) / 2
    P_minus = (I2 - A) / 2
    
    p_plus = np.real(np.trace(P_plus @ rho_A))
    p_minus = np.real(np.trace(P_minus @ rho_A))
    
    return p_plus, p_minus

def test_correlator_formula():
    """Verify correlator matches analytic formula for singlet."""
    print("PART 0: Correlator sanity check")
    
    rho = singlet()
    
    test_angles = [(0, 0), (0, np.pi/4), (np.pi/4, 0), (np.pi/2, np.pi/4)]
    all_ok = True
    
    for theta_A, theta_B in test_angles:
        E_num = correlator(rho, theta_A, theta_B)
        E_theory = -np.cos(theta_A - theta_B)
        ok = abs(E_num - E_theory) < 1e-10
        all_ok = all_ok and ok
        print(f"  E({theta_A:.4f}, {theta_B:.4f}) = {E_num:+.6f}, theory = {E_theory:+.6f} {'✓' if ok else '✗'}")
    
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok

def test_chsh_singlet():
    """CHSH violation for singlet with optimal angles."""
    print("PART A: Singlet CHSH")
    
    rho = singlet()
    
    # Optimal angles for S = 2√2:
    # a1 = 0, a2 = π/2, b1 = π/4, b2 = 3π/4
    # Then:
    # E(0, π/4) = -cos(-π/4) = -1/√2
    # E(0, 3π/4) = -cos(-3π/4) = +1/√2
    # E(π/2, π/4) = -cos(π/4) = -1/√2
    # E(π/2, 3π/4) = -cos(-π/4) = -1/√2
    # S = -1/√2 + 1/√2 + (-1/√2) - (-1/√2) = 0 ???
    #
    # Wait, let me recalculate with correct formula.
    # Standard optimal: a1=0, a2=π/2, b1=π/4, b2=-π/4
    # E(a1,b1) = -cos(0 - π/4) = -cos(-π/4) = -1/√2
    # E(a1,b2) = -cos(0 - (-π/4)) = -cos(π/4) = -1/√2
    # E(a2,b1) = -cos(π/2 - π/4) = -cos(π/4) = -1/√2
    # E(a2,b2) = -cos(π/2 - (-π/4)) = -cos(3π/4) = +1/√2
    # S = E11 + E12 + E21 - E22 = -1/√2 + (-1/√2) + (-1/√2) - (+1/√2) = -4/√2 = -2√2 ✓
    
    a1, a2 = 0, np.pi/2
    b1, b2 = np.pi/4, -np.pi/4
    
    E11 = correlator(rho, a1, b1)
    E12 = correlator(rho, a1, b2)
    E21 = correlator(rho, a2, b1)
    E22 = correlator(rho, a2, b2)
    
    print(f"  Angles: a1=0, a2=π/2, b1=π/4, b2=-π/4")
    print(f"  E(a1,b1) = {E11:+.6f}")
    print(f"  E(a1,b2) = {E12:+.6f}")
    print(f"  E(a2,b1) = {E21:+.6f}")
    print(f"  E(a2,b2) = {E22:+.6f}")
    
    S = E11 + E12 + E21 - E22
    S_expected = -2 * np.sqrt(2)
    
    print(f"  S = E11 + E12 + E21 - E22 = {S:+.6f}")
    print(f"  Expected: -2√2 = {S_expected:+.6f}")
    print(f"  |S| = {abs(S):.6f}, classical bound = 2.0")
    
    ok = abs(S - S_expected) < 1e-10
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_no_signalling():
    """Alice's marginals don't depend on Bob's setting."""
    print("PART B: No-signalling")
    
    rho = singlet()
    theta_A = np.pi / 6
    
    p_plus_reduced, p_minus_reduced = marginal_alice(rho, theta_A)
    
    print(f"  Alice θ_A = π/6")
    print(f"  From reduced ρ_A: P(+)={p_plus_reduced:.6f}, P(-)={p_minus_reduced:.6f}")
    
    # Verify by summing over Bob outcomes for various Bob settings
    bob_settings = [0, np.pi/4, np.pi/2, np.pi]
    
    A = spin_observable(theta_A)
    P_A_plus = (I2 + A) / 2
    P_A_minus = (I2 - A) / 2
    
    all_match = True
    print(f"  Summing over Bob outcomes:")
    for theta_B in bob_settings:
        B = spin_observable(theta_B)
        P_B_plus = (I2 + B) / 2
        P_B_minus = (I2 - B) / 2
        
        # P(a=+1) = P(a=+1, b=+1) + P(a=+1, b=-1)
        P_pp = np.kron(P_A_plus, P_B_plus)
        P_pm = np.kron(P_A_plus, P_B_minus)
        P_mp = np.kron(P_A_minus, P_B_plus)
        P_mm = np.kron(P_A_minus, P_B_minus)
        
        p_a_plus = np.real(np.trace(P_pp @ rho) + np.trace(P_pm @ rho))
        p_a_minus = np.real(np.trace(P_mp @ rho) + np.trace(P_mm @ rho))
        
        match = abs(p_a_plus - p_plus_reduced) < 1e-10
        all_match = all_match and match
        print(f"    θ_B={theta_B:.4f}: P(+)={p_a_plus:.6f} {'✓' if match else '✗'}")
    
    ok = all_match and abs(p_plus_reduced - 0.5) < 1e-10
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_product_state():
    """Product state satisfies classical bound."""
    print("PART C: Product state |00⟩")
    
    rho = product_00()
    a1, a2 = 0, np.pi/2
    b1, b2 = np.pi/4, -np.pi/4
    
    S = chsh(rho, a1, a2, b1, b2)
    
    print(f"  S = {S:+.6f}, |S| = {abs(S):.6f}")
    print(f"  Classical bound: 2.0")
    
    ok = abs(S) <= 2.0 + 1e-10
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_werner_threshold():
    """Werner states: violation iff p > 1/√2."""
    print("PART D: Werner states")
    
    a1, a2 = 0, np.pi/2
    b1, b2 = np.pi/4, -np.pi/4
    
    threshold = 1 / np.sqrt(2)
    test_ps = [0.0, 0.5, 0.7, threshold - 0.01, threshold + 0.01, 0.9, 1.0]
    
    print(f"  Threshold: 1/√2 ≈ {threshold:.6f}")
    print(f"  p        |S|       violates  expected")
    
    all_correct = True
    for p in test_ps:
        rho = werner(p)
        S = chsh(rho, a1, a2, b1, b2)
        violates = abs(S) > 2 + 1e-10
        expected = p > threshold + 1e-10
        correct = violates == expected
        all_correct = all_correct and correct
        print(f"  {p:.4f}   {abs(S):.6f}   {str(violates):5s}     {str(expected):5s}  {'✓' if correct else '✗'}")
    
    print(f"  Result: {'PASS' if all_correct else 'FAIL'}")
    return all_correct

def test_tsirelson():
    """Verify Tsirelson bound |S| ≤ 2√2."""
    print("PART E: Tsirelson bound")
    
    rho = singlet()
    tsirelson = 2 * np.sqrt(2)
    
    # Scan angles
    max_S = 0
    for a1 in np.linspace(0, np.pi, 8):
        for a2 in np.linspace(0, np.pi, 8):
            for b1 in np.linspace(0, np.pi, 8):
                for b2 in np.linspace(0, np.pi, 8):
                    S = abs(chsh(rho, a1, a2, b1, b2))
                    if S > max_S:
                        max_S = S
    
    print(f"  Max |S| found: {max_S:.6f}")
    print(f"  Tsirelson: 2√2 = {tsirelson:.6f}")
    
    ok = max_S <= tsirelson + 1e-6
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def main():
    print("=== T29_SV_bell_chsh ===")
    print()
    print("CCM interpretation:")
    print("  - Entangled Bulk state, no Facts until Commit")
    print("  - Spacelike Commits, order irrelevant (DI-D4)")
    print("  - Born weights give correlations exceeding classical bound")
    print("  - No-signalling from CPTP structure")
    print()
    
    ok_0 = test_correlator_formula()
    print()
    ok_A = test_chsh_singlet()
    print()
    ok_B = test_no_signalling()
    print()
    ok_C = test_product_state()
    print()
    ok_D = test_werner_threshold()
    print()
    ok_E = test_tsirelson()
    print()
    
    overall = ok_0 and ok_A and ok_B and ok_C and ok_D and ok_E
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  0. Correlator check:   {'PASS' if ok_0 else 'FAIL'}")
    print(f"  A. CHSH violation:     {'PASS' if ok_A else 'FAIL'}")
    print(f"  B. No-signalling:      {'PASS' if ok_B else 'FAIL'}")
    print(f"  C. Product ≤ 2:        {'PASS' if ok_C else 'FAIL'}")
    print(f"  D. Werner threshold:   {'PASS' if ok_D else 'FAIL'}")
    print(f"  E. Tsirelson bound:    {'PASS' if ok_E else 'FAIL'}")
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())