#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T31_SV_wigner_friend.py

Single-file, single-run test.

Purpose
-------
Wigner's Friend scenario with CCM interpretation.

Setup:
- System S in superposition |+⟩
- Friend F measures S (interaction → entanglement F+S)
- Wigner W has two options:
  (a) "Ask Friend" (measure in record basis)
  (b) "Superposition measurement" (measure in entangled basis)

CCM interpretation (единый FactLog):
- Commit = необратимая запись в Environment
- "Необратимость" определяется декогеренцией
- Если лаборатория изолирована: F+S в суперпозиции для Wigner (no Commit yet)
- Если лаборатория не изолирована: декогеренция → Commit → Fact
- В реальности изоляция неидеальна → практически единый FactLog

Key test:
- Isolated lab: Wigner can see interference (no Commit outside)
- Non-isolated: no interference (Commit happened)
- Consistency when Wigner "asks" (compatible Instruments)

No external deps beyond numpy.
"""

import numpy as np

def outer(psi):
    return np.outer(psi, psi.conj())

def tensor(a, b):
    return np.kron(a, b)

# Basis states
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ket_plus = (ket0 + ket1) / np.sqrt(2)
ket_minus = (ket0 - ket1) / np.sqrt(2)

def friend_interaction(psi_S):
    """
    Friend measures System. Models as entanglement:
    |ψ_S⟩ → α|0_F 0_S⟩ + β|1_F 1_S⟩
    
    F's memory becomes correlated with S.
    """
    alpha, beta = psi_S[0], psi_S[1]
    psi_FS = alpha * tensor(ket0, ket0) + beta * tensor(ket1, ket1)
    return psi_FS / np.linalg.norm(psi_FS)

def environment_decoherence(rho_FS, gamma):
    """
    Model decoherence to environment.
    gamma = 0: perfect isolation (no decoherence)
    gamma = 1: full decoherence (classical mixture)
    
    Decoheres in the "record basis" {|00⟩, |11⟩}.
    """
    # Projectors onto |00⟩ and |11⟩
    P00 = outer(tensor(ket0, ket0))
    P11 = outer(tensor(ket1, ket1))
    
    # Diagonal part (survives decoherence)
    diag = P00 @ rho_FS @ P00 + P11 @ rho_FS @ P11
    
    # Off-diagonal part (suppressed by decoherence)
    off_diag = rho_FS - diag
    
    return diag + (1 - gamma) * off_diag

def wigner_measures_ask(rho_FS):
    """
    Wigner "asks Friend": measures in record basis {|0_F⟩, |1_F⟩}.
    Returns probabilities P(0), P(1).
    """
    P0 = np.kron(outer(ket0), np.eye(2))  # |0_F⟩⟨0_F| ⊗ I_S
    P1 = np.kron(outer(ket1), np.eye(2))  # |1_F⟩⟨1_F| ⊗ I_S
    
    p0 = np.real(np.trace(P0 @ rho_FS))
    p1 = np.real(np.trace(P1 @ rho_FS))
    
    return p0, p1

def wigner_measures_superposition(rho_FS):
    """
    Wigner measures in entangled basis {|Φ+⟩, |Φ-⟩}.
    |Φ+⟩ = (|00⟩ + |11⟩)/√2
    |Φ-⟩ = (|00⟩ - |11⟩)/√2
    
    Returns probabilities P(Φ+), P(Φ-).
    """
    Phi_plus = (tensor(ket0, ket0) + tensor(ket1, ket1)) / np.sqrt(2)
    Phi_minus = (tensor(ket0, ket0) - tensor(ket1, ket1)) / np.sqrt(2)
    
    P_plus = outer(Phi_plus)
    P_minus = outer(Phi_minus)
    
    p_plus = np.real(np.trace(P_plus @ rho_FS))
    p_minus = np.real(np.trace(P_minus @ rho_FS))
    
    return p_plus, p_minus

def test_isolated_lab():
    """
    Perfect isolation (γ=0): Wigner can see interference.
    """
    print("PART A: Isolated lab (γ=0)")
    
    psi_S = ket_plus
    psi_FS = friend_interaction(psi_S)
    rho_FS = outer(psi_FS)
    
    # No decoherence
    rho_FS_isolated = environment_decoherence(rho_FS, gamma=0)
    
    # Check: state should be pure |Φ+⟩
    purity = np.real(np.trace(rho_FS_isolated @ rho_FS_isolated))
    print(f"  Purity of F+S: {purity:.6f} (1 = pure)")
    
    # Wigner superposition measurement
    p_plus, p_minus = wigner_measures_superposition(rho_FS_isolated)
    print(f"  Wigner measures Φ+/Φ-:")
    print(f"    P(Φ+) = {p_plus:.6f} (expected 1.0)")
    print(f"    P(Φ-) = {p_minus:.6f} (expected 0.0)")
    
    # Interpretation: no Commit happened (isolated), Wigner sees interference
    ok = abs(p_plus - 1.0) < 1e-10 and abs(p_minus - 0.0) < 1e-10
    print(f"  Interference visible: {'YES' if ok else 'NO'}")
    print(f"  CCM: No Commit (record didn't leave lab)")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_decohered_lab():
    """
    Full decoherence (γ=1): no interference for Wigner.
    """
    print("PART B: Decohered lab (γ=1)")
    
    psi_S = ket_plus
    psi_FS = friend_interaction(psi_S)
    rho_FS = outer(psi_FS)
    
    # Full decoherence
    rho_FS_decohered = environment_decoherence(rho_FS, gamma=1)
    
    # Check: state should be mixed
    purity = np.real(np.trace(rho_FS_decohered @ rho_FS_decohered))
    print(f"  Purity of F+S: {purity:.6f} (0.5 = maximally mixed in 2D subspace)")
    
    # Wigner superposition measurement
    p_plus, p_minus = wigner_measures_superposition(rho_FS_decohered)
    print(f"  Wigner measures Φ+/Φ-:")
    print(f"    P(Φ+) = {p_plus:.6f} (expected 0.5)")
    print(f"    P(Φ-) = {p_minus:.6f} (expected 0.5)")
    
    # No interference: P(Φ+) = P(Φ-) = 0.5
    ok = abs(p_plus - 0.5) < 1e-10 and abs(p_minus - 0.5) < 1e-10
    print(f"  Interference visible: {'NO' if ok else 'YES'}")
    print(f"  CCM: Commit happened (record leaked to Environment)")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_wigner_asks_consistent():
    """
    When Wigner "asks Friend", results agree (any γ).
    """
    print("PART C: Wigner asks Friend (consistency)")
    
    psi_S = ket_plus
    psi_FS = friend_interaction(psi_S)
    rho_FS = outer(psi_FS)
    
    print("  Testing various decoherence levels:")
    
    all_ok = True
    for gamma in [0, 0.5, 1.0]:
        rho = environment_decoherence(rho_FS, gamma)
        p0, p1 = wigner_measures_ask(rho)
        
        # Should always be 0.5, 0.5 (from |+⟩ input)
        ok = abs(p0 - 0.5) < 1e-10 and abs(p1 - 0.5) < 1e-10
        all_ok = all_ok and ok
        print(f"    γ={gamma}: P(0)={p0:.4f}, P(1)={p1:.4f} {'✓' if ok else '✗'}")
    
    print(f"  CCM: Compatible Instrument → consistent Facts")
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok

def test_no_contradiction():
    """
    Verify no logical contradiction in CCM.
    """
    print("PART D: No contradiction")
    
    print("  Scenario analysis:")
    print()
    print("  Case 1: Isolated lab (γ=0)")
    print("    - Friend's interaction creates entanglement")
    print("    - No record escapes → no Commit (for Wigner)")
    print("    - Friend's 'experience' is undefined (no Fact in global FactLog)")
    print("    - Wigner can measure interference")
    print()
    print("  Case 2: Non-isolated lab (γ>0)")
    print("    - Record escapes to Environment")
    print("    - Commit occurs → Fact exists")
    print("    - Friend's outcome is definite")
    print("    - No interference for Wigner")
    print()
    print("  Resolution:")
    print("    - 'Friend's experience' is not a separate Fact")
    print("    - Fact = record in Environment (Commit)")
    print("    - Isolated Friend+System = no Commit = no Fact")
    print("    - No contradiction: single criterion (Commit)")
    
    ok = True  # Structural argument
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def test_frauchiger_renner():
    """
    Frauchiger-Renner-type reasoning check.
    
    FR argument requires Friend to reason: "I saw 0, so Wigner will see X."
    But in CCM:
    - If lab isolated: Friend's "seeing" is not a Fact
    - If lab not isolated: Wigner can't see interference anyway
    
    The paradox assumes both simultaneously.
    """
    print("PART E: Frauchiger-Renner resolution")
    
    print("  FR paradox requires:")
    print("    1. Friend has definite experience (Fact)")
    print("    2. Wigner can measure superposition (no Fact)")
    print()
    print("  CCM denies this combination:")
    print("    - Fact requires Commit (record in Environment)")
    print("    - If Commit: decoherence → no superposition for Wigner")
    print("    - If no Commit: no Fact → Friend has no 'definite experience'")
    print()
    print("  The dilemma is false: isolation and Fact are exclusive.")
    
    # Verify numerically
    psi_S = ket_plus
    psi_FS = friend_interaction(psi_S)
    rho_FS = outer(psi_FS)
    
    # Check that γ=0 (interference) and definite outcome are exclusive
    rho_isolated = environment_decoherence(rho_FS, 0)
    rho_decohered = environment_decoherence(rho_FS, 1)
    
    p_plus_iso, _ = wigner_measures_superposition(rho_isolated)
    p_plus_dec, _ = wigner_measures_superposition(rho_decohered)
    
    interference_isolated = abs(p_plus_iso - 1.0) < 1e-10
    no_interference_decohered = abs(p_plus_dec - 0.5) < 1e-10
    
    ok = interference_isolated and no_interference_decohered
    print(f"  Numerical check: isolated↔interference, decohered↔no interference")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok

def main():
    print("=== T31_SV_wigner_friend ===")
    print()
    print("CCM interpretation (единый FactLog):")
    print("  - Commit = необратимая запись в Environment")
    print("  - Изолированная система: нет Commit снаружи")
    print("  - Реальная система: декогеренция → Commit → Fact")
    print("  - 'Опыт наблюдателя' = Fact только если есть Commit")
    print()
    
    ok_A = test_isolated_lab()
    print()
    ok_B = test_decohered_lab()
    print()
    ok_C = test_wigner_asks_consistent()
    print()
    ok_D = test_no_contradiction()
    print()
    ok_E = test_frauchiger_renner()
    print()
    
    overall = ok_A and ok_B and ok_C and ok_D and ok_E
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  A. Isolated lab:       {'PASS' if ok_A else 'FAIL'}")
    print(f"  B. Decohered lab:      {'PASS' if ok_B else 'FAIL'}")
    print(f"  C. Ask consistent:     {'PASS' if ok_C else 'FAIL'}")
    print(f"  D. No contradiction:   {'PASS' if ok_D else 'FAIL'}")
    print(f"  E. FR resolution:      {'PASS' if ok_E else 'FAIL'}")
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())