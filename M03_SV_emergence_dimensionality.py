#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M03_PLATEAU: Dimensionality via plateau detection.

Key insight: NN have EQUAL correlations (by symmetry).
Method: Count how many top correlations have same value (within tolerance).
This is k. Then d = k/2.
"""

import numpy as np

def make_1d(L):
    pos = [(x,) for x in range(L)]
    return pos, L, 1

def make_2d(L):
    pos = [(x, y) for y in range(L) for x in range(L)]
    return pos, L, 2

def make_3d(L):
    pos = [(x, y, z) for z in range(L) for y in range(L) for x in range(L)]
    return pos, L, 3

def distance_torus(p1, p2, L):
    """Manhattan distance on torus."""
    return sum(min(abs(a - b), L - abs(a - b)) for a, b in zip(p1, p2))

def build_correlation_matrix(positions, L, xi=1.5):
    """C(r) = exp(-r/xi), physically motivated."""
    n = len(positions)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r = distance_torus(positions[i], positions[j], L)
                C[i, j] = np.exp(-r / xi)
    return C

def detect_k_plateau(C, tol=0.01):
    """
    Detect k by counting plateau size.
    
    NN have identical |C| by symmetry. Count how many share
    the maximum correlation value (within tolerance).
    """
    n = C.shape[0]
    k_values = []
    
    for i in range(n):
        corrs = sorted([abs(C[i, j]) for j in range(n) if i != j], reverse=True)
        
        if len(corrs) < 2:
            continue
        
        # Find plateau: count values within tol of maximum
        max_c = corrs[0]
        k_i = sum(1 for c in corrs if abs(c - max_c) / max_c < tol)
        k_values.append(k_i)
    
    return np.mean(k_values), np.std(k_values)

def detect_k_gap(C):
    """
    Alternative: find largest GAP (absolute difference), not ratio.
    """
    n = C.shape[0]
    k_values = []
    
    for i in range(n):
        corrs = sorted([abs(C[i, j]) for j in range(n) if i != j], reverse=True)
        
        if len(corrs) < 2:
            continue
        
        # Find largest absolute gap
        gaps = [corrs[k] - corrs[k+1] for k in range(len(corrs)-1)]
        k_i = np.argmax(gaps) + 1
        k_values.append(k_i)
    
    return np.mean(k_values), np.std(k_values)

def show_structure(C, positions, L, dim):
    """Visualize correlation structure."""
    n = len(positions)
    corrs = []
    for j in range(1, n):
        r = distance_torus(positions[0], positions[j], L)
        corrs.append((abs(C[0, j]), r))
    corrs.sort(reverse=True)
    
    print(f"\n  Top correlations (site 0):")
    print(f"  {'Rank':<5} {'|C|':<12} {'Dist':<5}")
    print(f"  " + "-"*25)
    
    prev_c = None
    for rank, (c, r) in enumerate(corrs[:12], 1):
        gap_marker = ""
        if prev_c is not None:
            gap = prev_c - c
            if gap > 0.01:
                gap_marker = f" ← gap={gap:.4f}"
        print(f"  {rank:<5} {c:<12.5f} {r:<5}{gap_marker}")
        prev_c = c

def test(name, positions, L, dim, d_true):
    n = len(positions)
    k_true = 2 * d_true
    
    print(f"\n{'='*55}")
    print(f"{name}")
    print(f"{'='*55}")
    print(f"  Sites: {n}, d_true: {d_true}, k_true: {k_true}")
    
    C = build_correlation_matrix(positions, L)
    
    show_structure(C, positions, L, dim)
    
    # Method 1: Plateau
    k_plateau, k_std1 = detect_k_plateau(C)
    d_plateau = k_plateau / 2
    
    # Method 2: Gap
    k_gap, k_std2 = detect_k_gap(C)
    d_gap = k_gap / 2
    
    print(f"\n  Plateau method: k={k_plateau:.1f}±{k_std1:.1f}, d={d_plateau:.2f}")
    print(f"  Gap method:     k={k_gap:.1f}±{k_std2:.1f}, d={d_gap:.2f}")
    
    # Use plateau (more robust for symmetric lattices)
    error = abs(d_plateau - d_true)
    ok = error < 0.3
    print(f"\n  Final: d={d_plateau:.2f} (true={d_true}), error={error:.2f}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    
    return ok, d_plateau

def main():
    print("="*55)
    print("M03 PLATEAU: Dimensionality via plateau detection")
    print("="*55)
    print("\nMethod: Count NN with EQUAL correlation (symmetry).")
    print("This finds k directly, then d = k/2.")
    
    results = []
    
    # 1D
    pos, L, dim = make_1d(12)
    ok, d = test("1D Ring (L=12)", pos, L, dim, 1)
    results.append(("1D", ok, d, 1))
    
    # 2D
    pos, L, dim = make_2d(5)
    ok, d = test("2D Torus (5×5)", pos, L, dim, 2)
    results.append(("2D", ok, d, 2))
    
    # 3D — the key test!
    pos, L, dim = make_3d(4)
    ok, d = test("3D Torus (4×4×4)", pos, L, dim, 3)
    results.append(("3D", ok, d, 3))
    
    # 3D larger
    pos, L, dim = make_3d(6)
    ok, d = test("3D Torus (6×6×6)", pos, L, dim, 3)
    results.append(("3D L=6", ok, d, 3))
    
    # Negative control
    print(f"\n{'='*55}")
    print("NEGATIVE CONTROL")
    print("="*55)
    rng = np.random.default_rng(42)
    C_rand = rng.uniform(0, 1, (64, 64))
    C_rand = (C_rand + C_rand.T) / 2
    np.fill_diagonal(C_rand, 0)
    k_rand, _ = detect_k_plateau(C_rand, tol=0.05)
    print(f"  Random 64×64: k={k_rand:.1f}, d={k_rand/2:.1f}")
    neg_ok = k_rand < 3 or k_rand > 10  # Should not give sensible k=6
    print(f"  RESULT: {'PASS' if neg_ok else 'FAIL'} (expect no consistent k)")
    results.append(("NEG", neg_ok, None, None))
    
    # Summary
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    for name, ok, d_det, d_true in results:
        if d_true:
            print(f"  {name:<10}: d={d_det:.2f} (true={d_true}) {'PASS' if ok else 'FAIL'}")
        else:
            print(f"  {name:<10}: {'PASS' if ok else 'FAIL'}")
    
    physical = [ok for name, ok, _, _ in results if "NEG" not in name]
    all_pass = all(physical) and results[-1][1]
    
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    
    if all_pass:
        print("\n" + "*"*55)
        print("  EMERGENCE LAYER VALIDATED")
        print("  Dimension d = 1, 2, 3 extracted from correlations!")
        print("*"*55)
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    raise SystemExit(main())