#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E02_SV_gauge_invariance

Single-file, single-run test.

Claim
-----
Physical observables (B, holonomy) are gauge-invariant.
A_μ itself changes under gauge (not observable).

Test logic
----------
Part A: A changes under gauge (negative control)
Part B: B = ∇×A is invariant
Part C: ∮A·dl (closed loop) is invariant
Part D: ∫A·dl (open path) changes (negative control)
"""

import numpy as np


def A_field(x, y, gauge_chi=None):
    """Vector potential for uniform B_z = 1 (Landau gauge)."""
    Ax = 0.0
    Ay = x
    
    if gauge_chi is not None:
        Ax += gauge_chi * y
        Ay += gauge_chi * x
    
    return np.array([Ax, Ay, 0.0])


def B_field(x, y, gauge_chi=None):
    """Compute B = ∇×A numerically."""
    eps = 1e-8
    
    A_xp = A_field(x + eps, y, gauge_chi)
    A_xm = A_field(x - eps, y, gauge_chi)
    A_yp = A_field(x, y + eps, gauge_chi)
    A_ym = A_field(x, y - eps, gauge_chi)
    
    dAy_dx = (A_xp[1] - A_xm[1]) / (2 * eps)
    dAx_dy = (A_yp[0] - A_ym[0]) / (2 * eps)
    
    Bz = dAy_dx - dAx_dy
    return np.array([0, 0, Bz])


def line_integral_A(path, gauge_chi=None):
    """Compute ∫A·dl along path."""
    integral = 0.0
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        
        A = A_field(xm, ym, gauge_chi)
        dl = np.array([x1 - x0, y1 - y0, 0])
        
        integral += np.dot(A, dl)
    
    return integral


def main() -> int:
    print("=== E02_SV_gauge_invariance ===")
    print()
    
    # ========================================
    # Part A: A changes under gauge
    # ========================================
    print("-" * 60)
    print("PART A: A_μ changes under gauge (expected)")
    print("-" * 60)
    
    x, y = 2.0, 3.0
    A_orig = A_field(x, y, gauge_chi=None)
    A_trans = A_field(x, y, gauge_chi=0.5)
    
    diff = np.linalg.norm(A_trans - A_orig)
    ok_A = diff > 0.1
    
    print(f"  A (original)    = {A_orig}")
    print(f"  A (transformed) = {A_trans}")
    print(f"  ||ΔA|| = {diff:.6f}")
    print(f"  RESULT: {'PASS' if ok_A else 'FAIL'}")
    print()
    
    # ========================================
    # Part B: B is gauge-invariant
    # ========================================
    print("-" * 60)
    print("PART B: B field is gauge-invariant")
    print("-" * 60)
    
    B_orig = B_field(x, y, gauge_chi=None)
    B_trans = B_field(x, y, gauge_chi=0.5)
    
    diff_B = np.linalg.norm(B_trans - B_orig)
    ok_B = diff_B < 1e-6
    
    print(f"  B (original)    = {B_orig}")
    print(f"  B (transformed) = {B_trans}")
    print(f"  ||ΔB|| = {diff_B:.2e}")
    print(f"  RESULT: {'PASS' if ok_B else 'FAIL'}")
    print()
    
    # ========================================
    # Part C: Closed loop is invariant
    # ========================================
    print("-" * 60)
    print("PART C: ∮A·dl (closed loop) is gauge-invariant")
    print("-" * 60)
    
    N = 100
    L = 2.0
    path = []
    for i in range(N):
        path.append((i * L / N, 0))
    for i in range(N):
        path.append((L, i * L / N))
    for i in range(N):
        path.append((L - i * L / N, L))
    for i in range(N):
        path.append((0, L - i * L / N))
    path.append((0, 0))
    
    int_orig = line_integral_A(path, gauge_chi=None)
    int_trans = line_integral_A(path, gauge_chi=0.5)
    
    diff_int = abs(int_trans - int_orig)
    theory = L * L
    
    ok_C = diff_int < 1e-6 and abs(int_orig - theory) < 0.01
    
    print(f"  ∮A·dl (original)    = {int_orig:.6f}")
    print(f"  ∮A·dl (transformed) = {int_trans:.6f}")
    print(f"  ∮A·dl (theory)      = {theory:.6f}")
    print(f"  RESULT: {'PASS' if ok_C else 'FAIL'}")
    print()
    
    # ========================================
    # Part D: Open path changes
    # ========================================
    print("-" * 60)
    print("PART D: ∫A·dl (open path) changes (negative control)")
    print("-" * 60)
    
    open_path = [(0, 0), (1, 0), (1, 1), (2, 1)]
    
    open_orig = line_integral_A(open_path, gauge_chi=None)
    open_trans = line_integral_A(open_path, gauge_chi=0.5)
    
    diff_open = abs(open_trans - open_orig)
    ok_D = diff_open > 0.1
    
    print(f"  ∫A·dl (original)    = {open_orig:.6f}")
    print(f"  ∫A·dl (transformed) = {open_trans:.6f}")
    print(f"  RESULT: {'PASS' if ok_D else 'FAIL'}")
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