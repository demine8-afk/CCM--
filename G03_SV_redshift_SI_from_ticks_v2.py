#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G03_SV_redshift_SI_from_ticks_v2

Fixes:
- GPS PASS criterion compares to GR exact (Schwarzschild), not weak-field approx.
- weak-field remains as printed sanity/comparison.

Clock model: record-limited sharp threshold, deep saturation.
Tick-rate r_tick ∝ Omega_local ∝ T(x).
"""

import math

# SI constants
c = 299792458.0
G = 6.67430e-11
SECONDS_PER_DAY = 86400.0

# Sun
M_sun = 1.98892e30
R_sun = 6.9634e8

# Earth/GPS
M_earth = 5.9722e24
R_earth = 6.371e6
h_GPS = 20200e3
R_GPS = R_earth + h_GPS


def Phi_newton(M, r):
    return -G * M / r


def lapse_from_Phi(Phi):
    # layer choice: T = sqrt(1 + 2Phi/c^2) = sqrt(1 - r_s/r) for point mass
    arg = 1.0 + 2.0 * Phi / (c * c)
    return math.sqrt(max(arg, 1e-30))


def lapse_GR_exact_point_mass(M, r):
    # Schwarzschild: sqrt(1 - r_s/r), r_s = 2GM/c^2
    r_s = 2.0 * G * M / (c * c)
    return math.sqrt(1.0 - r_s / r)


def tick_rate(Omega_local, delta=1e-4, kappa=200.0):
    denom = 2.0 * math.acos(math.sqrt(delta)) + 1.0 / kappa
    return Omega_local / denom


def main():
    print("=== G03_SV_redshift_SI_from_ticks_v2 ===")
    delta = 1e-4
    kappa = 200.0
    Omega0 = 1.0
    print(f"delta={delta:.1e}, kappa={kappa:.1f}, Omega0={Omega0:.3f}")
    print()

    # ---------------- Solar: surface -> infinity ----------------
    Phi_sun = Phi_newton(M_sun, R_sun)
    T_emit = lapse_from_Phi(Phi_sun)
    T_obs = 1.0

    r_emit = tick_rate(T_emit * Omega0, delta, kappa)
    r_obs  = tick_rate(T_obs  * Omega0, delta, kappa)

    ratio = r_obs / r_emit
    z_ticks = ratio - 1.0

    # GR exact
    r_s = 2.0 * G * M_sun / (c * c)
    z_GR_exact = 1.0 / math.sqrt(1.0 - r_s / R_sun) - 1.0
    err_solar = abs(z_ticks - z_GR_exact)

    print("--- SOLAR ---")
    print(f"T_emit = {T_emit:.12f}")
    print(f"tick-rate ratio r_obs/r_emit = {ratio:.12f}")
    print(f"z_ticks     = {z_ticks:.12e}")
    print(f"z_GR_exact  = {z_GR_exact:.12e}")
    print(f"abs error   = {err_solar:.3e}")
    print()

    # ---------------- GPS: grav only ----------------
    Phi_earth = Phi_newton(M_earth, R_earth)
    Phi_gps   = Phi_newton(M_earth, R_GPS)

    T_earth = lapse_from_Phi(Phi_earth)
    T_gps   = lapse_from_Phi(Phi_gps)

    r_earth = tick_rate(T_earth * Omega0, delta, kappa)
    r_gps   = tick_rate(T_gps   * Omega0, delta, kappa)

    ratio_g = r_gps / r_earth
    frac = ratio_g - 1.0
    delta_us_ticks = frac * SECONDS_PER_DAY * 1e6

    # GR exact (point-mass Schwarzschild)
    T_earth_GR = lapse_GR_exact_point_mass(M_earth, R_earth)
    T_gps_GR   = lapse_GR_exact_point_mass(M_earth, R_GPS)
    ratio_GR_exact = T_gps_GR / T_earth_GR
    frac_GR_exact = ratio_GR_exact - 1.0
    delta_us_GR_exact = frac_GR_exact * SECONDS_PER_DAY * 1e6

    # GR weak-field approx (sanity only)
    frac_GR_wf = (Phi_gps - Phi_earth) / (c * c)
    delta_us_GR_wf = frac_GR_wf * SECONDS_PER_DAY * 1e6

    print("--- GPS (grav only) ---")
    print(f"T_earth (CCM) = {T_earth:.15f}")
    print(f"T_gps   (CCM) = {T_gps:.15f}")
    print(f"tick-rate ratio r_gps/r_earth = {ratio_g:.15f}")
    print(f"Δt/day (ticks)    = {delta_us_ticks:+.6f} μs")
    print(f"Δt/day (GR exact) = {delta_us_GR_exact:+.6f} μs")
    print(f"Δt/day (GR wf)    = {delta_us_GR_wf:+.6f} μs")
    print()

    err_gps_exact = abs(delta_us_ticks - delta_us_GR_exact)
    print(f"abs error vs GR exact = {err_gps_exact:.3e} μs")

    ok_solar = err_solar < 1e-15
    ok_gps   = err_gps_exact < 1e-12  # microseconds, should be ~0 here

    print("\n=== VERDICT ===")
    print(f"solar: {'PASS' if ok_solar else 'FAIL'}")
    print(f"gps  : {'PASS' if ok_gps else 'FAIL'}")
    print(f"OVERALL: {'PASS' if (ok_solar and ok_gps) else 'FAIL'}")


if __name__ == "__main__":
    main()