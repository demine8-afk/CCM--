#!/usr/bin/env python3
"""
Z01_SV_quantum_zeno_ccm.py

Quantum Zeno effect в CCM: сравнение с идеальным (λ → ∞).

CCM-СПЕЦИФИЧНОСТЬ:
- Zeno ограничен record speed limit: λ ≤ ω_record
- При λ ~ Ω: неполное замораживание
- При λ >> Ω: приближение к идеальному Zeno

Setup:
- Двухуровневая система: |0⟩ (survival), |1⟩ (decay)
- Bulk: H = (ℏΩ/2)σ_x → Rabi oscillations
- Survival weight: w(τ) = cos²(Ωτ/2)
- Commit: Poisson rate λ
- При Commit: survival → reset to |0⟩, decay → stop

Аналитика:
P_one = ∫₀^∞ λ exp(-λτ) cos²(Ωτ/2) dτ = (2λ² + Ω²)/(2(λ² + Ω²))

При N Commits (в среднем N = λT):
P_all ≈ (P_one)^N = exp(-λT · |ln(P_one)|)

Эффективный decay rate:
λ_eff = λ · |ln(P_one)|

CCM prediction:
- λ → ∞: λ_eff → 0 (Zeno works)
- λ ~ Ω: λ_eff ~ Ω (no Zeno)
- λ → 0: λ_eff → λ·ln(2) (averaging)

Тест: Monte Carlo vs аналитика, разные λ/Ω.
"""

import numpy as np

# ========================================
# SI Constants (для reference)
# ========================================
HBAR = 1.054571817e-34  # J·s

# ========================================
# Model parameters
# ========================================
OMEGA = 2 * np.pi * 1e9  # 1 GHz Rabi frequency (rad/s)
T_TOTAL = 100 / OMEGA     # Total time: 100 Rabi periods

# λ/Ω ratios to scan
LAMBDA_RATIOS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

N_RUNS = 10000  # Monte Carlo runs per λ


def survival_weight(tau, omega):
    """Born weight for survival: |⟨0|ψ(τ)⟩|² = cos²(Ωτ/2)."""
    return np.cos(omega * tau / 2)**2


def p_one_survival_analytic(lam, omega):
    """
    Probability of survival at single Commit:
    P = ∫₀^∞ λ exp(-λτ) cos²(Ωτ/2) dτ
      = (2λ² + Ω²) / (2(λ² + Ω²))
    """
    return (2 * lam**2 + omega**2) / (2 * (lam**2 + omega**2))


def effective_decay_rate(lam, omega):
    """
    Effective decay rate in Zeno regime:
    λ_eff = λ · |ln(P_one)|
    """
    p_one = p_one_survival_analytic(lam, omega)
    if p_one >= 1.0:
        return 0.0
    return lam * abs(np.log(p_one))


def p_survival_analytic(lam, omega, t_total):
    """
    Probability of all-survival until t_total:
    P ≈ exp(-λ_eff · t_total)
    """
    lam_eff = effective_decay_rate(lam, omega)
    return np.exp(-lam_eff * t_total)


def monte_carlo_zeno(lam, omega, t_total, n_runs):
    """
    Monte Carlo simulation of Zeno in CCM.
    
    Returns: fraction of runs where system survived all Commits.
    """
    survival_count = 0
    
    for _ in range(n_runs):
        t = 0.0
        tau_since_last = 0.0  # time since last Commit (or start)
        survived = True
        
        while t < t_total and survived:
            # Next Commit: exponential waiting time
            dt = np.random.exponential(1.0 / lam)
            t += dt
            tau_since_last += dt
            
            if t >= t_total:
                # No Commit before end — still survived
                break
            
            # Commit happened: check survival
            w_surv = survival_weight(tau_since_last, omega)
            
            if np.random.random() < w_surv:
                # Survival: reset to |0⟩
                tau_since_last = 0.0
            else:
                # Decay: stop
                survived = False
        
        if survived:
            survival_count += 1
    
    return survival_count / n_runs


def main():
    print("="*70)
    print("Z01: QUANTUM ZENO EFFECT IN CCM")
    print("="*70)
    print()
    print(f"Ω = 2π × {OMEGA/(2*np.pi)/1e9:.1f} GHz")
    print(f"T = {T_TOTAL * OMEGA:.0f} / Ω = {T_TOTAL*1e9:.2f} ns")
    print(f"Monte Carlo runs per λ: {N_RUNS}")
    print()
    
    # Part A: Analytic P_one vs λ/Ω
    print("-"*70)
    print("PART A: Single-Commit survival probability P_one(λ)")
    print("-"*70)
    print(f"{'λ/Ω':>10} {'P_one':>12} {'1-P_one':>12}")
    print("-"*40)
    
    for ratio in LAMBDA_RATIOS:
        lam = ratio * OMEGA
        p_one = p_one_survival_analytic(lam, OMEGA)
        print(f"{ratio:>10.1f} {p_one:>12.6f} {1-p_one:>12.6f}")
    
    print()
    print("Limits:")
    print("  λ → ∞: P_one → 1 (Zeno)")
    print("  λ → 0: P_one → 1/2 (average over oscillations)")
    print()
    
    # Part B: Monte Carlo vs Analytic
    print("-"*70)
    print("PART B: Survival probability P(T) — Monte Carlo vs Analytic")
    print("-"*70)
    print(f"{'λ/Ω':>10} {'P_MC':>12} {'P_analytic':>12} {'|Δ|':>10} {'λ_eff/Ω':>10}")
    print("-"*60)
    
    results = []
    all_pass = True
    
    for ratio in LAMBDA_RATIOS:
        lam = ratio * OMEGA
        
        # Monte Carlo
        p_mc = monte_carlo_zeno(lam, OMEGA, T_TOTAL, N_RUNS)
        
        # Analytic
        p_ana = p_survival_analytic(lam, OMEGA, T_TOTAL)
        
        # Effective decay rate
        lam_eff = effective_decay_rate(lam, OMEGA) / OMEGA
        
        # Error
        delta = abs(p_mc - p_ana)
        
        # Expected statistical error ~ sqrt(p(1-p)/N)
        stat_err = np.sqrt(p_mc * (1 - p_mc) / N_RUNS) if 0 < p_mc < 1 else 0.01
        
        ok = delta < 5 * stat_err + 0.02  # tolerance
        if not ok:
            all_pass = False
        
        print(f"{ratio:>10.1f} {p_mc:>12.4f} {p_ana:>12.4f} {delta:>10.4f} {lam_eff:>10.4f}")
        results.append((ratio, p_mc, p_ana, lam_eff))
    
    print()
    
    # Part C: CCM-specific prediction
    print("-"*70)
    print("PART C: CCM-SPECIFIC PREDICTION — Zeno limited by record speed")
    print("-"*70)
    print()
    print("Key insight:")
    print("  Standard QM: λ → ∞ achievable, P → 1 (complete Zeno)")
    print("  CCM: λ ≤ ω_record (record speed limit)")
    print()
    print("If ω_record ~ Ω (fast system, slow detector):")
    print(f"  λ_max = Ω → P_one = {p_one_survival_analytic(OMEGA, OMEGA):.4f}")
    print(f"  λ_eff/Ω = {effective_decay_rate(OMEGA, OMEGA)/OMEGA:.4f}")
    print(f"  P(T=100/Ω) = {p_survival_analytic(OMEGA, OMEGA, T_TOTAL):.4e}")
    print("  → Zeno FAILS for fast oscillators with slow detectors!")
    print()
    print("If ω_record >> Ω (fast detector):")
    print(f"  λ = 100Ω → P_one = {p_one_survival_analytic(100*OMEGA, OMEGA):.6f}")
    print(f"  P(T=100/Ω) = {p_survival_analytic(100*OMEGA, OMEGA, T_TOTAL):.4f}")
    print("  → Zeno works")
    print()
    
    # Part D: Scaling law
    print("-"*70)
    print("PART D: Scaling law — λ_eff vs λ")
    print("-"*70)
    print()
    print("At λ >> Ω: λ_eff ≈ Ω²/(2λ) → 0 (Zeno)")
    print("At λ << Ω: λ_eff ≈ λ·ln(2) (no Zeno, averaging)")
    print("At λ = Ω:  λ_eff ≈ 0.29·Ω (transition)")
    print()
    
    # Verify scaling
    lam_high = 100 * OMEGA
    lam_eff_high = effective_decay_rate(lam_high, OMEGA)
    lam_eff_predicted = OMEGA**2 / (2 * lam_high)
    
    print(f"Check λ >> Ω: λ_eff = {lam_eff_high/OMEGA:.4f}Ω, predicted Ω/(2·100) = {lam_eff_predicted/OMEGA:.4f}Ω")
    
    scaling_ok = abs(lam_eff_high - lam_eff_predicted) / OMEGA < 0.01
    print(f"Scaling match: {'PASS' if scaling_ok else 'FAIL'}")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("1. CCM reproduces Zeno effect in limit λ → ∞")
    print("2. CCM-SPECIFIC: Zeno limited by record speed limit λ ≤ ω_record")
    print("3. TESTABLE: Fast system (large Ω) + slow detector → incomplete Zeno")
    print()
    print(f"Monte Carlo vs Analytic: {'PASS' if all_pass else 'FAIL'}")
    print(f"Scaling law: {'PASS' if scaling_ok else 'FAIL'}")
    print()
    
    if all_pass and scaling_ok:
        print("="*70)
        print("Z01 QUANTUM ZENO CCM: PASS")
        print("="*70)
        return 0
    else:
        print("="*70)
        print("Z01 QUANTUM ZENO CCM: FAIL")
        print("="*70)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())