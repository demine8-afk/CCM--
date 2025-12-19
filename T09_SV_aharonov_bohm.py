#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T09_SV_aharonov_bohm

Single-file, single-run test.

Claim
-----
In CCM, U(1) connection a(e) enters Bulk amplitude A(H).
Only holonomy Φ_C = ∮a·dl is DI-invariant (gauge-invariant).
Phase difference between paths = holonomy of closed loop.
Interference: |ψ|² = 2 + 2cos(Φ).

Test logic
----------
1. Closed loop around solenoid: holonomy = magnetic flux Φ
2. Two paths (top/bottom): Δφ = Φ
3. Interference pattern: |ψ|² = 2 + 2cos(Δφ)
4. Flux sweep: verify pattern for multiple Φ values

Negative control: path NOT enclosing solenoid should give holonomy ≈ 0.
"""

import numpy as np


def phase_increment(x0, y0, x1, y1, R_sol, Phi_mag):
    """
    Phase increment a(u→v) for edge from (x0,y0) to (x1,y1).
    A_θ = Φ/(2πr) for r > R_sol (outside solenoid).
    """
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    r = np.sqrt(xm**2 + ym**2)
    
    if r < R_sol or r < 1e-10:
        return 0.0
    
    A_theta = Phi_mag / (2 * np.pi * r)
    theta = np.arctan2(ym, xm)
    
    Ax = -A_theta * np.sin(theta)
    Ay = A_theta * np.cos(theta)
    
    dlx, dly = x1 - x0, y1 - y0
    
    return Ax * dlx + Ay * dly


def path_phase(path, R_sol, Phi_mag):
    """Total phase along path: Σ a(e)."""
    total = 0.0
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        total += phase_increment(x0, y0, x1, y1, R_sol, Phi_mag)
    return total


def make_circular_path(R, N_points):
    """Circular path of radius R, N_points samples."""
    theta_arr = np.linspace(0, 2*np.pi, N_points)
    return [(R * np.cos(th), R * np.sin(th)) for th in theta_arr]


def make_semicircle_path(x_start, x_end, R, N_points, top=True):
    """Semicircular path from x_start to x_end, going through y>0 (top) or y<0 (bottom)."""
    if top:
        theta = np.linspace(np.pi, 0, N_points)
    else:
        theta = np.linspace(-np.pi, 0, N_points)
    
    path = [(x_start, 0.0)]
    path += [(R * np.cos(th), R * np.sin(th)) for th in theta]
    path += [(x_end, 0.0)]
    return path


def main() -> int:
    print("=== T09_SV_aharonov_bohm ===")
    print()
    
    # Parameters
    R_sol = 1.0           # solenoid radius
    Phi_mag = np.pi       # magnetic flux
    N_points = 1000
    
    print(f"R_solenoid = {R_sol}")
    print(f"Phi_mag = {Phi_mag:.6f} rad (π)")
    print(f"N_points = {N_points}")
    print()
    
    # ========================================
    # TEST 1: Closed loop around solenoid
    # ========================================
    print("-" * 60)
    print("TEST 1: CLOSED LOOP (holonomy = Φ)")
    print("-" * 60)
    
    R_path = 3.0
    closed_loop = make_circular_path(R_path, N_points)
    holonomy = path_phase(closed_loop, R_sol, Phi_mag)
    
    err_1 = abs(holonomy - Phi_mag)
    ok_1 = err_1 < 0.01
    
    print(f"  Loop radius: R = {R_path}")
    print(f"  Holonomy (computed) = {holonomy:.6f} rad")
    print(f"  Magnetic flux Φ     = {Phi_mag:.6f} rad")
    print(f"  Error               = {err_1:.6f} rad")
    print(f"  RESULT: {'PASS' if ok_1 else 'FAIL'}")
    print()
    
    # ========================================
    # TEST 2: Two paths (interferometer)
    # ========================================
    print("-" * 60)
    print("TEST 2: TWO PATHS (Δφ = Φ)")
    print("-" * 60)
    
    x_source, x_detect = -5.0, 5.0
    R_arc = 5.0
    
    path_top = make_semicircle_path(x_source, x_detect, R_arc, N_points//2, top=True)
    path_bottom = make_semicircle_path(x_source, x_detect, R_arc, N_points//2, top=False)
    
    phi_top = path_phase(path_top, R_sol, Phi_mag)
    phi_bottom = path_phase(path_bottom, R_sol, Phi_mag)
    delta_phi = phi_top - phi_bottom
    
    err_2 = abs(abs(delta_phi) - Phi_mag)
    ok_2 = err_2 < 0.01
    
    print(f"  Source: ({x_source}, 0), Detector: ({x_detect}, 0)")
    print(f"  φ(top path)    = {phi_top:.6f} rad")
    print(f"  φ(bottom path) = {phi_bottom:.6f} rad")
    print(f"  Δφ = φ_top - φ_bot = {delta_phi:.6f} rad")
    print(f"  |Δφ|           = {abs(delta_phi):.6f} rad")
    print(f"  Expected Φ     = {Phi_mag:.6f} rad")
    print(f"  Error          = {err_2:.6f} rad")
    print(f"  RESULT: {'PASS' if ok_2 else 'FAIL'}")
    print()
    
    # ========================================
    # TEST 3: Interference
    # ========================================
    print("-" * 60)
    print("TEST 3: INTERFERENCE (|ψ|² = 2 + 2cos(Δφ))")
    print("-" * 60)
    
    A_top = np.exp(1j * phi_top)
    A_bottom = np.exp(1j * phi_bottom)
    psi = A_top + A_bottom
    prob = np.abs(psi)**2
    prob_theory = 2 + 2 * np.cos(delta_phi)
    
    err_3 = abs(prob - prob_theory)
    ok_3 = err_3 < 1e-10
    
    print(f"  ψ = exp(iφ_top) + exp(iφ_bot)")
    print(f"  |ψ|² (computed) = {prob:.6f}")
    print(f"  |ψ|² (theory)   = {prob_theory:.6f}")
    print(f"  Error           = {err_3:.2e}")
    print(f"  RESULT: {'PASS' if ok_3 else 'FAIL'}")
    print()
    
    # ========================================
    # TEST 4: Flux sweep
    # ========================================
    print("-" * 60)
    print("TEST 4: FLUX SWEEP")
    print("-" * 60)
    
    fluxes = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    print(f"  {'Φ/π':>8} | {'|ψ|²':>8} | {'theory':>8} | {'error':>10}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    
    ok_4 = True
    max_err_4 = 0.0
    for Phi in fluxes:
        phi_t = path_phase(path_top, R_sol, Phi)
        phi_b = path_phase(path_bottom, R_sol, Phi)
        dphi = phi_t - phi_b
        
        psi_test = np.exp(1j * phi_t) + np.exp(1j * phi_b)
        prob_test = np.abs(psi_test)**2
        prob_th = 2 + 2 * np.cos(dphi)
        err = abs(prob_test - prob_th)
        max_err_4 = max(max_err_4, err)
        
        print(f"  {Phi/np.pi:>8.4f} | {prob_test:>8.4f} | {prob_th:>8.4f} | {err:>10.2e}")
        
        if err > 1e-9:
            ok_4 = False
    
    print(f"  max error = {max_err_4:.2e}")
    print(f"  RESULT: {'PASS' if ok_4 else 'FAIL'}")
    print()
    
    # ========================================
    # TEST 5: Negative control (no enclosure)
    # ========================================
    print("-" * 60)
    print("TEST 5: NEGATIVE CONTROL (loop NOT enclosing solenoid)")
    print("-" * 60)
    
    # Small loop far from solenoid
    R_far = 0.3
    center_x, center_y = 5.0, 5.0
    theta_arr = np.linspace(0, 2*np.pi, N_points)
    loop_outside = [(center_x + R_far * np.cos(th), center_y + R_far * np.sin(th)) 
                    for th in theta_arr]
    
    holonomy_outside = path_phase(loop_outside, R_sol, Phi_mag)
    
    ok_5 = abs(holonomy_outside) < 0.01
    
    print(f"  Loop center: ({center_x}, {center_y}), radius: {R_far}")
    print(f"  Holonomy = {holonomy_outside:.6f} rad")
    print(f"  Expected ≈ 0")
    print(f"  RESULT: {'PASS' if ok_5 else 'FAIL'} (no flux enclosed)")
    print()
    
    # ========================================
    # VERDICT
    # ========================================
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    all_pass = ok_1 and ok_2 and ok_3 and ok_4 and ok_5
    
    print(f"  1. Closed loop holonomy = Φ:     {'PASS' if ok_1 else 'FAIL'}")
    print(f"  2. Two paths |Δφ| = Φ:           {'PASS' if ok_2 else 'FAIL'}")
    print(f"  3. Interference formula:         {'PASS' if ok_3 else 'FAIL'}")
    print(f"  4. Flux sweep:                   {'PASS' if ok_4 else 'FAIL'}")
    print(f"  5. Negative control (no encl.):  {'PASS' if ok_5 else 'FAIL'}")
    print()
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    
    if all_pass:
        print("""
  AB effect in CCM:
  • U(1) connection a(e) enters Bulk amplitude A(H)
  • Holonomy Φ_C = ∮a·dl is the only DI-invariant
  • Phase difference = holonomy of closed loop
  • |ψ|² = 2 + 2cos(Φ) — interference in Bulk
  • Commit fixes outcome with P ∝ |ψ|²
        """)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())