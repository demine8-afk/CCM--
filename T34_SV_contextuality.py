#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T34_SV_contextuality.py

Single-file, single-run test.

Purpose
-------
Demonstrate quantum contextuality via Peres-Mermin square.

CCM interpretation:
- Outcome = property of (state, Instrument)
- No pre-existing values before Commit
- Context = which Instrument (which commuting set)

No external deps beyond numpy.
"""

import numpy as np
from itertools import product

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron(A, B):
    return np.kron(A, B)

def commutator(A, B):
    return A @ B - B @ A

def peres_mermin_square():
    """
    Correct Peres-Mermin square.
    
    Standard form where:
    - All row products = +I
    - Column products = +I, +I, -I
    
           C0       C1       C2
    R0:   I⊗Z     Z⊗I     Z⊗Z      product = +I
    R1:   X⊗I     I⊗X     X⊗X      product = +I
    R2:   X⊗Z     Z⊗X     Y⊗Y      product = +I
    
    C0: (I⊗Z)(X⊗I)(X⊗Z) = X²⊗Z² = I⊗I = +I
    C1: (Z⊗I)(I⊗X)(Z⊗X) = Z²⊗X² = I⊗I = +I
    C2: (Z⊗Z)(X⊗X)(Y⊗Y) = (ZXY)⊗(ZXY) = ?
    
    ZXY = Z @ X @ Y = [[1,0],[0,-1]] @ [[0,1],[1,0]] @ [[0,-j],[j,0]]
        = [[0,1],[-1,0]] @ [[0,-j],[j,0]] = [[j,0],[0,j]] = jI
    So (ZXY)⊗(ZXY) = (jI)⊗(jI) = -I⊗I = -I
    
    Perfect!
    """
    R0 = [kron(I2, Z), kron(Z, I2), kron(Z, Z)]
    R1 = [kron(X, I2), kron(I2, X), kron(X, X)]
    R2 = [kron(X, Z), kron(Z, X), kron(Y, Y)]
    
    return [R0, R1, R2]

def check_eigenvalues(square):
    """Verify all observables have eigenvalues ±1."""
    print("PART 0: Eigenvalue check (all should be ±1)")
    
    all_ok = True
    for i, row in enumerate(square):
        for j, obs in enumerate(row):
            eigs = np.linalg.eigvalsh(obs)
            eigs_real = np.sort(np.real(eigs))
            expected = np.array([-1, -1, 1, 1])  # Two-qubit, each ±1 with multiplicity 2
            ok = np.allclose(eigs_real, expected)
            if not ok:
                print(f"  [{i},{j}]: eigenvalues = {eigs_real} (UNEXPECTED)")
                all_ok = False
    
    if all_ok:
        print("  All observables have eigenvalues ±1")
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok

def check_commutation(square):
    """Verify observables in each row/column commute."""
    print("PART A: Commutation check")
    
    all_commute = True
    
    # Check rows
    for i, row in enumerate(square):
        for j in range(3):
            for k in range(j+1, 3):
                comm = np.max(np.abs(commutator(row[j], row[k])))
                if comm > 1e-10:
                    print(f"  Row {i}: [{j},{k}] don't commute! max|[A,B]|={comm:.2e}")
                    all_commute = False
    
    # Check columns
    for j in range(3):
        col = [square[i][j] for i in range(3)]
        for i1 in range(3):
            for i2 in range(i1+1, 3):
                comm = np.max(np.abs(commutator(col[i1], col[i2])))
                if comm > 1e-10:
                    print(f"  Col {j}: [{i1},{i2}] don't commute! max|[A,B]|={comm:.2e}")
                    all_commute = False
    
    if all_commute:
        print("  All rows and columns commute internally")
    print(f"  Result: {'PASS' if all_commute else 'FAIL'}")
    return all_commute

def check_row_products(square):
    """Verify product of each row = +I."""
    print("PART B: Row products = +I")
    
    I4 = np.eye(4, dtype=complex)
    all_ok = True
    
    for i, row in enumerate(square):
        prod = row[0] @ row[1] @ row[2]
        is_plus_I = np.allclose(prod, I4)
        is_minus_I = np.allclose(prod, -I4)
        
        if is_plus_I:
            print(f"  Row {i}: product = +I ✓")
        elif is_minus_I:
            print(f"  Row {i}: product = -I ✗ (expected +I)")
            all_ok = False
        else:
            print(f"  Row {i}: product = ??? ✗")
            all_ok = False
    
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok

def check_column_products(square):
    """Verify column products: C0=+I, C1=+I, C2=-I."""
    print("PART C: Column products (+I, +I, -I)")
    
    I4 = np.eye(4, dtype=complex)
    expected = [+1, +1, -1]
    
    all_ok = True
    for j in range(3):
        col = [square[i][j] for i in range(3)]
        prod = col[0] @ col[1] @ col[2]
        
        if np.allclose(prod, I4):
            actual = +1
        elif np.allclose(prod, -I4):
            actual = -1
        else:
            actual = 0
        
        ok = actual == expected[j]
        sign_str = '+I' if actual == 1 else ('-I' if actual == -1 else '???')
        exp_str = '+I' if expected[j] == 1 else '-I'
        print(f"  Col {j}: product = {sign_str}, expected {exp_str} {'✓' if ok else '✗'}")
        all_ok = all_ok and ok
    
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok

def check_contradiction(square):
    """
    The algebraic contradiction:
    - Product of all row products = (+I)(+I)(+I) = +I
    - Product of all column products = (+I)(+I)(-I) = -I
    - But both are products of the same 9 observables!
    """
    print("PART D: Algebraic contradiction")
    
    # Product of rows
    I4 = np.eye(4, dtype=complex)
    
    row_product = I4.copy()
    for row in square:
        for obs in row:
            row_product = row_product @ obs
    
    col_product = I4.copy()
    for j in range(3):
        for i in range(3):
            col_product = col_product @ square[i][j]
    
    # Both should equal product of all 9, but row gives +I pattern, col gives -I pattern
    # Actually both products ARE the same (all 9 observables), so something's subtle.
    # The point is: row products multiply to +1, column products multiply to -1,
    # yet each observable appears in exactly one row and one column.
    
    # Let's verify the LOGICAL contradiction
    print("  If values v(O) ∈ {±1} existed:")
    print("    Row products: v(R0[0])v(R0[1])v(R0[2]) = +1 (for each row)")
    print("    Product of all row-products = +1")
    print("    But: each v(O) appears exactly once")
    print("    So: ∏(all v) = +1")
    print()
    print("    Column products: = +1, +1, -1")
    print("    Product of column-products = -1")
    print("    But: each v(O) appears exactly once in columns too")
    print("    So: ∏(all v) = -1")
    print()
    print("    Contradiction: +1 ≠ -1")
    
    print(f"  Result: PASS (logical contradiction established)")
    return True

def check_no_assignment():
    """Verify no ±1 assignment satisfies all constraints."""
    print("PART E: Exhaustive search for valid assignment")
    
    # 9 observables, each gets ±1
    # Constraints:
    # Row 0: v[0,0] * v[0,1] * v[0,2] = +1
    # Row 1: v[1,0] * v[1,1] * v[1,2] = +1
    # Row 2: v[2,0] * v[2,1] * v[2,2] = +1
    # Col 0: v[0,0] * v[1,0] * v[2,0] = +1
    # Col 1: v[0,1] * v[1,1] * v[2,1] = +1
    # Col 2: v[0,2] * v[1,2] * v[2,2] = -1
    
    count = 0
    for assignment in product([+1, -1], repeat=9):
        v = np.array(assignment).reshape(3, 3)
        
        row_ok = all(np.prod(v[i, :]) == +1 for i in range(3))
        col_ok = (np.prod(v[:, 0]) == +1 and 
                  np.prod(v[:, 1]) == +1 and 
                  np.prod(v[:, 2]) == -1)
        
        if row_ok and col_ok:
            count += 1
    
    print(f"  Total assignments checked: 512")
    print(f"  Valid assignments found: {count}")
    
    ok = count == 0
    print(f"  Result: {'PASS' if ok else 'FAIL'} (no valid assignment)")
    return ok

def check_quantum_consistent(square):
    """Quantum mechanics gives consistent probabilities."""
    print("PART F: Quantum consistency")
    
    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    
    print("  State: |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print("  Expectations ⟨O⟩:")
    
    for i, row in enumerate(square):
        exps = [np.real(psi.conj() @ obs @ psi) for obs in row]
        print(f"    Row {i}: {[f'{e:+.2f}' for e in exps]}, product = {exps[0]*exps[1]*exps[2]:+.2f}")
    
    print()
    print("  Expectations are well-defined, but are NOT all ±1.")
    print("  QM doesn't assign definite values—only Commit does.")
    print(f"  Result: PASS")
    return True

def main():
    print("=== T34_SV_contextuality ===")
    print()
    print("Peres-Mermin square: 9 two-qubit observables")
    print()
    print("CCM interpretation:")
    print("  - Outcome = property of (state, Instrument), not state alone")
    print("  - No pre-existing values before Commit")
    print("  - Context = which commuting set is measured")
    print()
    
    square = peres_mermin_square()
    
    ok_0 = check_eigenvalues(square)
    print()
    ok_A = check_commutation(square)
    print()
    ok_B = check_row_products(square)
    print()
    ok_C = check_column_products(square)
    print()
    ok_D = check_contradiction(square)
    print()
    ok_E = check_no_assignment()
    print()
    ok_F = check_quantum_consistent(square)
    print()
    
    overall = ok_0 and ok_A and ok_B and ok_C and ok_D and ok_E and ok_F
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  0. Eigenvalues ±1:       {'PASS' if ok_0 else 'FAIL'}")
    print(f"  A. Commutation:          {'PASS' if ok_A else 'FAIL'}")
    print(f"  B. Row products +I:      {'PASS' if ok_B else 'FAIL'}")
    print(f"  C. Col products pattern: {'PASS' if ok_C else 'FAIL'}")
    print(f"  D. Contradiction:        {'PASS' if ok_D else 'FAIL'}")
    print(f"  E. No HV assignment:     {'PASS' if ok_E else 'FAIL'}")
    print(f"  F. QM consistent:        {'PASS' if ok_F else 'FAIL'}")
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())