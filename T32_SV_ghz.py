#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T32_SV_ghz.py

Single-file, single-run test.

Purpose
-------
GHZ state: three-particle entanglement with "all-or-nothing" correlations.

|GHZ⟩ = (|000⟩ + |111⟩)/√2

Key properties:
1. XXX measurement: always even parity (+++, +--, -+-, --+)
2. XYY, YXY, YYX: always odd parity
3. Product of all four = -1 (QM) vs +1 (local HV) → contradiction without inequalities

CCM interpretation:
- GHZ is Bulk state
- Three spacelike Commits
- Born weights give correlations
- No assignment of pre-existing values consistent with all results

No external deps beyond numpy.
"""

import numpy as np
from itertools import product

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def ghz_state():
    """
    |GHZ⟩ = (|000⟩ + |111⟩)/√2
    Returns state vector (8,).
    """
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1  # |000⟩
    psi[7] = 1  # |111⟩
    return psi / np.sqrt(2)

def kron3(A, B, C):
    """Kronecker product of three matrices."""
    return np.kron(np.kron(A, B), C)

def expectation(psi, Op):
    """⟨ψ|Op|ψ⟩"""
    return np.real(np.vdot(psi, Op @ psi))

def measure_xxx(psi):
    """Expectation of X⊗X⊗X"""
    Op = kron3(X, X, X)
    return expectation(psi, Op)

def measure_xyy(psi):
    """Expectation of X⊗Y⊗Y"""
    Op = kron3(X, Y, Y)
    return expectation(psi, Op)

def measure_yxy(psi):
    """Expectation of Y⊗X⊗Y"""
    Op = kron3(Y, X, Y)
    return expectation(psi, Op)

def measure_yyx(psi):
    """Expectation of Y⊗Y⊗X"""
    Op = kron3(Y, Y, X)
    return expectation(psi, Op)

def parity_distribution(psi, Op):
    """
    Compute probability of each parity outcome.
    Op has eigenvalues ±1. Return P(+1), P(-1).
    """
    # Project onto +1 and -1 eigenspaces
    P_plus = (np.eye(8) + Op) / 2
    P_minus = (np.eye(8) - Op) / 2
    
    p_plus = np.real(np.vdot(psi, P_plus @ psi))
    p_minus = np.real(np.vdot(psi, P_minus @ psi))
    
    return p_plus, p_minus

def test_xxx_parity():
    """XXX: always +1 parity (even number of -1 outcomes)"""
    print("PART A: XXX measurement")
    
    psi = ghz_state()
    exp_xxx = measure_xxx(psi)
    p_plus, p_minus = parity_distribution(psi, kron3(X, X, X))
    
    print(f"  ⟨XXX⟩ = {exp_xxx:+.6f} (expected +1)")
    print(f"  P(parity +1) = {p_plus:.6f}")
    print(f"  P(parity -1) = {p_minus:.6f}")
    
    ok = abs(exp_xxx - 1.0) < 1e-10
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_xyy_yxy_yyx_parity():
    """XYY, YXY, YYX: always -1 parity (odd number of -1 outcomes)"""
    print("PART B: XYY, YXY, YYX measurements")
    
    psi = ghz_state()
    
    results = []
    for name, Op in [("XYY", kron3(X, Y, Y)), 
                      ("YXY", kron3(Y, X, Y)), 
                      ("YYX", kron3(Y, Y, X))]:
        exp_val = expectation(psi, Op)
        p_plus, p_minus = parity_distribution(psi, Op)
        print(f"  ⟨{name}⟩ = {exp_val:+.6f} (expected -1)")
        results.append(abs(exp_val + 1.0) < 1e-10)
    
    ok = all(results)
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_ghz_contradiction():
    """
    GHZ contradiction with local hidden variables.
    
    If outcomes were pre-determined:
    - XXX = +1 means x1*x2*x3 = +1
    - XYY = -1 means x1*y2*y3 = -1
    - YXY = -1 means y1*x2*y3 = -1
    - YYX = -1 means y1*y2*x3 = -1
    
    Multiply all four:
    (x1*x2*x3) * (x1*y2*y3) * (y1*x2*y3) * (y1*y2*x3) 
    = x1² * x2² * x3² * y1² * y2² * y3²
    = +1 (since all squares)
    
    But QM predicts: (+1)*(-1)*(-1)*(-1) = -1
    
    Contradiction! No local HV model possible.
    """
    print("PART C: GHZ contradiction")
    
    psi = ghz_state()
    
    exp_xxx = measure_xxx(psi)
    exp_xyy = measure_xyy(psi)
    exp_yxy = measure_yxy(psi)
    exp_yyx = measure_yyx(psi)
    
    qm_product = exp_xxx * exp_xyy * exp_yxy * exp_yyx
    hv_product = 1.0  # Must be +1 for any local HV assignment
    
    print(f"  QM: ⟨XXX⟩·⟨XYY⟩·⟨YXY⟩·⟨YYX⟩ = ({exp_xxx:+.0f})·({exp_xyy:+.0f})·({exp_yxy:+.0f})·({exp_yyx:+.0f}) = {qm_product:+.0f}")
    print(f"  Local HV: product of squares = +1")
    print(f"  QM ≠ HV: {qm_product:.0f} ≠ {hv_product:.0f}")
    
    ok = abs(qm_product - (-1.0)) < 1e-10
    print(f"  Result: {'PASS' if ok else 'FAIL'} (GHZ contradiction demonstrated)")
    return ok

def test_individual_outcomes():
    """
    Individual measurement outcomes are random (50/50).
    Only correlations are constrained.
    """
    print("PART D: Individual outcomes random")
    
    psi = ghz_state()
    
    # Measure first qubit in X basis
    P0_x = (np.eye(2) + X) / 2  # |+⟩⟨+|
    P0_full = kron3(P0_x, I2, I2)
    
    p_plus = np.real(np.vdot(psi, P0_full @ psi))
    
    print(f"  P(first qubit = +1 in X basis) = {p_plus:.6f} (expected 0.5)")
    
    ok = abs(p_plus - 0.5) < 1e-10
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def main():
    print("=== T32_SV_ghz ===")
    print()
    print("|GHZ⟩ = (|000⟩ + |111⟩)/√2")
    print()
    print("CCM interpretation:")
    print("  - GHZ is Bulk state (no Facts yet)")
    print("  - Three spacelike Commits possible")
    print("  - Born weights give perfect correlations")
    print("  - No pre-existing values (GHZ contradiction)")
    print()
    
    ok_A = test_xxx_parity()
    print()
    ok_B = test_xyy_yxy_yyx_parity()
    print()
    ok_C = test_ghz_contradiction()
    print()
    ok_D = test_individual_outcomes()
    print()
    
    overall = ok_A and ok_B and ok_C and ok_D
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())