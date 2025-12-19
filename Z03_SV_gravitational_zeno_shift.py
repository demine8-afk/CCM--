#!/usr/bin/env python3
"""
Z03_SV_gravitational_zeno_shift.py

Гравитационный сдвиг Zeno/anti-Zeno кривой.

CCM PREDICTION:
- λ_local(x) = T(x) · λ_I
- В гравитационном потенциале T(x) < 1
- Эффективное λ/Ω сдвигается
- Zeno efficiency зависит от высоты!

Setup:
- Два детектора: на Земле и на орбите GPS
- Одинаковый λ_I (одинаковые детекторы)
- Разные T(x) → разные λ_local

Это пересечение Gravity layer и Core (Commit law).
"""

import numpy as np

# Physical constants (SI)
G = 6.67430e-11      # m³/(kg·s²)
C = 299792458.0      # m/s
M_EARTH = 5.972e24   # kg
R_EARTH = 6.371e6    # m
H_GPS = 20200e3      # m (GPS orbit altitude)

OMEGA = 2 * np.pi * 1e9  # 1 GHz system
N_RUNS = 5000


def lapse_weak_field(r, M=M_EARTH):
    """T(r) = sqrt(1 + 2Φ/c²), Φ = -GM/r"""
    Phi = -G * M / r
    return np.sqrt(1 + 2 * Phi / C**2)


def p_one(lam, omega):
    """P(survival at one Commit)"""
    return (2 * lam**2 + omega**2) / (2 * (lam**2 + omega**2))


def lambda_eff(lam, omega):
    """Effective decay rate"""
    p = p_one(lam, omega)
    if p >= 1.0:
        return 0.0
    return lam * abs(np.log(p))


def mean_decay_time_mc(lam, omega, n_runs):
    """Monte Carlo: mean time to decay."""
    times = []
    for _ in range(n_runs):
        t, tau = 0.0, 0.0
        while True:
            dt = np.random.exponential(1.0 / lam)
            t += dt
            tau += dt
            w = np.cos(omega * tau / 2)**2
            if np.random.random() < w:
                tau = 0.0
            else:
                times.append(t)
                break
    return np.mean(times)


def main():
    print("="*70)
    print("Z03: GRAVITATIONAL ZENO SHIFT")
    print("="*70)
    print()
    
    # Locations
    r_surface = R_EARTH
    r_gps = R_EARTH + H_GPS
    
    T_surface = lapse_weak_field(r_surface)
    T_gps = lapse_weak_field(r_gps)
    
    print("PART A: Gravitational lapse at different heights")
    print("-"*70)
    print(f"Earth surface: r = {r_surface/1e6:.3f} Mm, T = {T_surface:.10f}")
    print(f"GPS orbit:     r = {r_gps/1e6:.3f} Mm, T = {T_gps:.10f}")
    print(f"ΔT/T = {(T_gps - T_surface)/T_surface:.3e}")
    print()
    
    # Same detector, different locations
    # Tune λ_I so that at GPS orbit we're at anti-Zeno peak
    lambda_I = OMEGA  # detector tuned to λ_I = Ω at infinity
    
    lambda_surface = T_surface * lambda_I
    lambda_gps = T_gps * lambda_I
    
    print("PART B: Local Commit rates")
    print("-"*70)
    print(f"λ_I = Ω (detector tuned to system frequency)")
    print(f"λ_local(surface) / Ω = {lambda_surface/OMEGA:.10f}")
    print(f"λ_local(GPS) / Ω     = {lambda_gps/OMEGA:.10f}")
    print(f"Δλ/λ = {(lambda_gps - lambda_surface)/lambda_surface:.3e}")
    print()
    
    # Effective decay rates
    print("PART C: Effective decay rates (analytic)")
    print("-"*70)
    
    leff_surface = lambda_eff(lambda_surface, OMEGA)
    leff_gps = lambda_eff(lambda_gps, OMEGA)
    
    print(f"λ_eff(surface) / Ω = {leff_surface/OMEGA:.10f}")
    print(f"λ_eff(GPS) / Ω     = {leff_gps/OMEGA:.10f}")
    print(f"Δλ_eff/λ_eff = {(leff_gps - leff_surface)/leff_surface:.3e}")
    print()
    
    # Near the peak, derivative is ~0, so effect is second order
    # Let's check at a point away from peak
    
    print("PART D: Enhanced effect away from peak")
    print("-"*70)
    
    # Tune detector so λ/Ω = 2 at GPS (on the Zeno slope)
    lambda_I_v2 = 2 * OMEGA
    
    lambda_surface_v2 = T_surface * lambda_I_v2
    lambda_gps_v2 = T_gps * lambda_I_v2
    
    leff_surface_v2 = lambda_eff(lambda_surface_v2, OMEGA)
    leff_gps_v2 = lambda_eff(lambda_gps_v2, OMEGA)
    
    print(f"λ_I = 2Ω (on Zeno slope)")
    print(f"λ_local(surface) / Ω = {lambda_surface_v2/OMEGA:.10f}")
    print(f"λ_local(GPS) / Ω     = {lambda_gps_v2/OMEGA:.10f}")
    print(f"λ_eff(surface) / Ω   = {leff_surface_v2/OMEGA:.10f}")
    print(f"λ_eff(GPS) / Ω       = {leff_gps_v2/OMEGA:.10f}")
    print(f"Δλ_eff/λ_eff = {(leff_gps_v2 - leff_surface_v2)/leff_surface_v2:.3e}")
    print()
    
    # Monte Carlo verification
    print("PART E: Monte Carlo verification")
    print("-"*70)
    
    # Use larger effect for numerical verification: scale up ΔT
    # Fictional "strong gravity" scenario
    T_low = 0.99   # strong gravity
    T_high = 1.00  # weak gravity
    
    lambda_low = T_low * OMEGA
    lambda_high = T_high * OMEGA
    
    leff_low_ana = lambda_eff(lambda_low, OMEGA) / OMEGA
    leff_high_ana = lambda_eff(lambda_high, OMEGA) / OMEGA
    
    # MC
    t_low = mean_decay_time_mc(lambda_low, OMEGA, N_RUNS)
    t_high = mean_decay_time_mc(lambda_high, OMEGA, N_RUNS)
    
    leff_low_mc = 1.0 / (t_low * OMEGA)
    leff_high_mc = 1.0 / (t_high * OMEGA)
    
    print(f"Strong gravity (T=0.99): λ_eff/Ω = {leff_low_mc:.4f} (ana: {leff_low_ana:.4f})")
    print(f"Weak gravity (T=1.00):   λ_eff/Ω = {leff_high_mc:.4f} (ana: {leff_high_ana:.4f})")
    
    # Check direction: lower T → lower λ → different λ_eff
    # Near peak, effect is small. Let's verify at T=0.5 vs T=1.0
    
    print()
    print("PART F: Large gravity differential (T=0.5 vs T=1.0)")
    print("-"*70)
    
    T_extreme = 0.5
    lambda_extreme = T_extreme * OMEGA
    
    leff_extreme_ana = lambda_eff(lambda_extreme, OMEGA) / OMEGA
    t_extreme = mean_decay_time_mc(lambda_extreme, OMEGA, N_RUNS)
    leff_extreme_mc = 1.0 / (t_extreme * OMEGA)
    
    print(f"T=0.5: λ/Ω = 0.5, λ_eff/Ω = {leff_extreme_mc:.4f} (ana: {leff_extreme_ana:.4f})")
    print(f"T=1.0: λ/Ω = 1.0, λ_eff/Ω = {leff_high_mc:.4f} (ana: {leff_high_ana:.4f})")
    print()
    
    # At λ/Ω = 0.5, we're BELOW the peak → lower λ_eff
    # At λ/Ω = 1.0, we're AT the peak → higher λ_eff
    
    shift_ok = leff_extreme_mc < leff_high_mc * 0.95  # at least 5% lower
    
    print(f"Gravity shifts λ_eff: {'PASS' if shift_ok else 'FAIL'}")
    print()
    
    # Interpretation
    print("="*70)
    print("CCM PREDICTION: GRAVITATIONAL ZENO SHIFT")
    print("="*70)
    print()
    print("Mechanism:")
    print("  λ_local(x) = T(x) · λ_I")
    print("  Lower T(x) → lower effective λ/Ω")
    print("  Position on Zeno curve shifts with gravity")
    print()
    print("Consequence:")
    print("  Same detector at different heights → different decay efficiency")
    print("  Deep in gravity well: shifted toward anti-Zeno or averaging regime")
    print("  High altitude: closer to intended λ/Ω ratio")
    print()
    print("Magnitude (Earth surface vs GPS):")
    print(f"  ΔT/T ≈ {(T_gps - T_surface)/T_surface:.1e}")
    print(f"  Effect on λ_eff: same order (enhanced on Zeno slope)")
    print()
    print("TESTABLE (in principle):")
    print("  Precision Zeno experiment at different altitudes")
    print("  Or near massive object vs far from it")
    print()
    
    if shift_ok:
        print("Z03 GRAVITATIONAL ZENO SHIFT: PASS")
        return 0
    else:
        print("Z03 GRAVITATIONAL ZENO SHIFT: FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())