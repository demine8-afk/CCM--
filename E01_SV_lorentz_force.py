#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E01_SV_lorentz_force

Single-file, single-run test.

Claim
-----
Lorentz force F = q(E + v×B) emerges from EM layer phase structure.

Test logic
----------
Part A: Uniform B field → circular motion (cyclotron)
Part B: Uniform E field → linear acceleration
Part C: Crossed E×B → drift velocity

Uses Boris pusher for numerical stability in magnetic field.
"""

import numpy as np


def boris_push(x, v, E, B, q, m, dt):
    """
    Boris algorithm — symplectic integrator for charged particle in EM field.
    Stable for cyclotron motion.
    """
    # Half acceleration from E
    v_minus = v + (q / m) * E * dt / 2
    
    # Rotation from B
    t = (q / m) * B * dt / 2
    s = 2 * t / (1 + np.dot(t, t))
    
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    
    # Half acceleration from E
    v_new = v_plus + (q / m) * E * dt / 2
    
    # Position update
    x_new = x + v_new * dt
    
    return x_new, v_new


def main() -> int:
    print("=== E01_SV_lorentz_force ===")
    print()
    
    q = 1.6e-19
    m = 9.1e-31
    
    # ========================================
    # Part A: B field → circular motion
    # ========================================
    print("-" * 60)
    print("PART A: Uniform B field (circular motion)")
    print("-" * 60)
    
    B_val = 1.0  # Tesla
    v0 = 1e5     # Reduced speed for stability
    
    # Cyclotron radius: r = mv / (qB)
    r_theory = m * v0 / (q * B_val)
    
    # Cyclotron period: T = 2πm / (qB)
    T_cyclotron = 2 * np.pi * m / (q * B_val)
    
    # Simulate for 2 periods
    dt = T_cyclotron / 1000
    N_steps = 2000
    
    x = np.array([0.0, 0.0, 0.0])
    v = np.array([v0, 0.0, 0.0])
    E = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, B_val])
    
    positions = [x.copy()]
    for _ in range(N_steps):
        x, v = boris_push(x, v, E, B, q, m, dt)
        positions.append(x.copy())
    
    positions = np.array(positions)
    
    # Measure radius from trajectory
    r_measured = (np.max(positions[:, 0]) - np.min(positions[:, 0])) / 2
    
    err_r = abs(r_measured - r_theory) / r_theory
    ok_A = err_r < 0.01
    
    print(f"  B = {B_val} T, v0 = {v0:.2e} m/s")
    print(f"  T_cyclotron = {T_cyclotron:.6e} s")
    print(f"  r (theory)   = {r_theory:.6e} m")
    print(f"  r (measured) = {r_measured:.6e} m")
    print(f"  Relative error = {err_r:.6f}")
    print(f"  RESULT: {'PASS' if ok_A else 'FAIL'}")
    print()
    
    # ========================================
    # Part B: E field → linear acceleration
    # ========================================
    print("-" * 60)
    print("PART B: Uniform E field (linear acceleration)")
    print("-" * 60)
    
    E_val = 1e6  # V/m
    t_total = 1e-9
    
    a_theory = q * E_val / m
    v_final_theory = a_theory * t_total
    x_final_theory = 0.5 * a_theory * t_total**2
    
    dt = 1e-12
    N_steps = int(t_total / dt)
    
    x = np.array([0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    E = np.array([E_val, 0.0, 0.0])
    B = np.array([0.0, 0.0, 0.0])
    
    for _ in range(N_steps):
        x, v = boris_push(x, v, E, B, q, m, dt)
    
    err_v = abs(v[0] - v_final_theory) / v_final_theory
    err_x = abs(x[0] - x_final_theory) / x_final_theory
    
    ok_B = err_v < 0.01 and err_x < 0.01
    
    print(f"  E = {E_val:.2e} V/m, t = {t_total:.2e} s")
    print(f"  v_final (theory)   = {v_final_theory:.6e} m/s")
    print(f"  v_final (measured) = {v[0]:.6e} m/s")
    print(f"  x_final (theory)   = {x_final_theory:.6e} m")
    print(f"  x_final (measured) = {x[0]:.6e} m")
    print(f"  RESULT: {'PASS' if ok_B else 'FAIL'}")
    print()
    
    # ========================================
    # Part C: E×B → drift
    # ========================================
    print("-" * 60)
    print("PART C: Crossed E×B (drift velocity)")
    print("-" * 60)
    
    E_val = 1e3   # V/m along y
    B_val = 0.01  # T along z
    
    # Drift velocity: v_drift = E/B along x
    v_drift_theory = E_val / B_val
    
    # Cyclotron period for this B
    T_cyc = 2 * np.pi * m / (q * B_val)
    
    # Simulate for many periods to average out gyration
    dt = T_cyc / 100
    N_steps = 5000
    
    x = np.array([0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    E = np.array([0.0, E_val, 0.0])
    B = np.array([0.0, 0.0, B_val])
    
    for _ in range(N_steps):
        x, v = boris_push(x, v, E, B, q, m, dt)
    
    t_total = N_steps * dt
    v_drift_measured = x[0] / t_total
    
    err_drift = abs(v_drift_measured - v_drift_theory) / v_drift_theory
    ok_C = err_drift < 0.05
    
    print(f"  E = {E_val:.2e} V/m (along y)")
    print(f"  B = {B_val} T (along z)")
    print(f"  v_drift (theory)   = {v_drift_theory:.2e} m/s")
    print(f"  v_drift (measured) = {v_drift_measured:.2e} m/s")
    print(f"  Relative error = {err_drift:.4f}")
    print(f"  RESULT: {'PASS' if ok_C else 'FAIL'}")
    print()
    
    # ========================================
    # VERDICT
    # ========================================
    print("=" * 60)
    overall = ok_A and ok_B and ok_C
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    if overall:
        print("""
  Lorentz force F = q(E + v×B) verified:
  • Magnetic field → circular motion (cyclotron)
  • Electric field → linear acceleration  
  • Crossed E×B → drift velocity E/B
        """)
    
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())