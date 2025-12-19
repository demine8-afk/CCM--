#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T30_SV_delayed_choice.py

Single-file, single-run test.

Purpose
-------
Wheeler's delayed choice experiment in CCM framework.

Setup:
- Photon enters Mach-Zehnder interferometer
- "Choice" to insert/remove second beam-splitter made AFTER photon passes first BS
- But BEFORE Commit (detector click)

CCM interpretation:
- Until Commit, everything is Bulk (unitary evolution)
- "Choice" changes the Instrument (which POVM is applied)
- No retrocausality: photon has no "path" until Commit
- Bulk contains superposition; Commit creates Fact

This test verifies:
1. Open interferometer: which-path info, no interference
2. Closed interferometer: interference pattern
3. "Choice" timing doesn't matter if before Commit
4. No hidden "path" variable needed

No external deps beyond numpy.
"""

import numpy as np

def beam_splitter():
    """50-50 beam splitter: |0⟩ → (|0⟩ + i|1⟩)/√2"""
    return np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)

def phase_shift(phi):
    """Phase shift in path 1"""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

def mz_open(phi):
    """
    Mach-Zehnder WITHOUT second beam splitter.
    Input |0⟩ → measure which path.
    """
    BS = beam_splitter()
    P = phase_shift(phi)
    # After first BS and phase: BS @ |0⟩, then phase
    state = P @ BS @ np.array([1, 0], dtype=complex)
    return state

def mz_closed(phi):
    """
    Mach-Zehnder WITH second beam splitter.
    Input |0⟩ → interference at output.
    """
    BS = beam_splitter()
    P = phase_shift(phi)
    # BS @ P @ BS @ |0⟩
    state = BS @ P @ BS @ np.array([1, 0], dtype=complex)
    return state

def probabilities(state):
    """Born probabilities for detector 0 and 1"""
    return np.abs(state[0])**2, np.abs(state[1])**2

def test_open_interferometer():
    """
    Without second BS: no interference, P(0) = P(1) = 0.5 for any phase.
    """
    print("PART A: Open interferometer (which-path)")
    
    phases = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    all_ok = True
    
    print("  Phase  |   P(D0)   |   P(D1)")
    for phi in phases:
        state = mz_open(phi)
        p0, p1 = probabilities(state)
        ok = abs(p0 - 0.5) < 1e-10 and abs(p1 - 0.5) < 1e-10
        all_ok = all_ok and ok
        print(f"  {phi:5.3f}  |  {p0:.6f}  |  {p1:.6f}  {'✓' if ok else '✗'}")
    
    print(f"  Result: {'PASS' if all_ok else 'FAIL'} (no interference)")
    return all_ok

def test_closed_interferometer():
    """
    With second BS: interference, P(0) and P(1) depend on phase.
    """
    print("PART B: Closed interferometer (interference)")
    
    # At phi = 0: constructive at D0
    # At phi = π: constructive at D1
    test_cases = [
        (0, 0.0, 1.0),           # All to D1 (check: BS is symmetric)
        (np.pi, 1.0, 0.0),       # All to D0
        (np.pi/2, 0.5, 0.5),     # Equal split
    ]
    
    all_ok = True
    print("  Phase  |   P(D0)   |   P(D1)   | Expected")
    for phi, exp0, exp1 in test_cases:
        state = mz_closed(phi)
        p0, p1 = probabilities(state)
        ok = abs(p0 - exp0) < 1e-10 and abs(p1 - exp1) < 1e-10
        all_ok = all_ok and ok
        print(f"  {phi:5.3f}  |  {p0:.6f}  |  {p1:.6f}  | ({exp0:.1f}, {exp1:.1f}) {'✓' if ok else '✗'}")
    
    print(f"  Result: {'PASS' if all_ok else 'FAIL'} (interference pattern)")
    return all_ok

def test_delayed_choice():
    """
    Key test: "choice" made after photon passes first BS.
    
    In CCM:
    - After first BS, Bulk state is superposition
    - No Commit yet → no Fact about "which path"
    - "Choice" selects Instrument (open vs closed POVM)
    - Commit happens at detector → creates Fact
    
    Result: same predictions as "normal" choice timing.
    """
    print("PART C: Delayed choice equivalence")
    
    phi = np.pi / 3  # Arbitrary phase
    
    # "Early" choice: decide before photon enters
    state_open_early = mz_open(phi)
    state_closed_early = mz_closed(phi)
    
    # "Delayed" choice: decide after first BS
    # In CCM, this is identical because:
    # - State after first BS is intermediate Bulk state
    # - "Choice" determines which unitary to apply next
    # - No Commit → no Fact → no difference
    
    # Simulate "delayed": state after first BS
    BS = beam_splitter()
    P = phase_shift(phi)
    state_after_first_BS = P @ BS @ np.array([1, 0], dtype=complex)
    
    # Now "choose" open (no second BS) or closed (second BS)
    state_open_delayed = state_after_first_BS  # Direct to detectors
    state_closed_delayed = BS @ state_after_first_BS  # Through second BS
    
    # Compare
    ok_open = np.allclose(state_open_early, state_open_delayed)
    ok_closed = np.allclose(state_closed_early, state_closed_delayed)
    
    print(f"  Open: early == delayed: {'PASS' if ok_open else 'FAIL'}")
    print(f"  Closed: early == delayed: {'PASS' if ok_closed else 'FAIL'}")
    
    return ok_open and ok_closed

def test_no_hidden_path():
    """
    Negative control: hidden "path" variable would give wrong predictions.
    
    If photon "really took path 0 or 1" after first BS:
    - With second BS: each path → 50/50 at output
    - Overall: 50/50 regardless of phase
    
    But QM (and CCM) predicts interference → contradiction.
    """
    print("PART D: Hidden path variable (negative control)")
    
    phi = 0  # Should give P(D1) = 1 in closed config
    
    # Hidden variable prediction (if path is real)
    # Path 0: BS gives 50/50 at output
    # Path 1: BS gives 50/50 at output  
    # Mixture: 50/50 regardless of phase
    hv_p0, hv_p1 = 0.5, 0.5
    
    # QM/CCM prediction
    state = mz_closed(phi)
    qm_p0, qm_p1 = probabilities(state)
    
    print(f"  Hidden variable: P(D0)={hv_p0}, P(D1)={hv_p1}")
    print(f"  QM/CCM:          P(D0)={qm_p0:.4f}, P(D1)={qm_p1:.4f}")
    
    # They differ → hidden path model fails
    differs = abs(qm_p0 - hv_p0) > 0.1 or abs(qm_p1 - hv_p1) > 0.1
    print(f"  Predictions differ: {'PASS' if differs else 'FAIL'} (HV model rejected)")
    
    return differs

def main():
    print("=== T30_SV_delayed_choice ===")
    print()
    print("CCM interpretation:")
    print("  - Bulk evolves unitarily until Commit")
    print("  - 'Choice' selects Instrument, not 'reality'")
    print("  - No Fact exists until detector Commit")
    print("  - No retrocausality needed")
    print()
    
    ok_A = test_open_interferometer()
    print()
    ok_B = test_closed_interferometer()
    print()
    ok_C = test_delayed_choice()
    print()
    ok_D = test_no_hidden_path()
    print()
    
    overall = ok_A and ok_B and ok_C and ok_D
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())