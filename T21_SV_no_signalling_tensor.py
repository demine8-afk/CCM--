#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T21_SV_no_signalling_tensor

Single-file, single-run test.

Claim
-----
For a bipartite system AB, local operations on A (CPTP, no postselection) do not change rho_B:
  rho_B  = Tr_A(rho_AB)
  rho_B' = Tr_A( (Phi_A ⊗ I)(rho_AB) )  == rho_B

Includes a negative control with an intentionally WRONG "trace" implementation,
which must show signalling-like changes.

No external deps beyond numpy.
"""

import numpy as np

def dagger(M):
    return M.conj().T

def fro_norm(M):
    return float(np.linalg.norm(M, ord='fro'))

def kron(A, B):
    return np.kron(A, B)

def partial_trace_A(rho_AB, dA, dB):
    # Tr_A over first subsystem
    rho_AB = rho_AB.reshape(dA, dB, dA, dB)
    rho_B = np.zeros((dB, dB), dtype=complex)
    for a in range(dA):
        rho_B += rho_AB[a, :, a, :]
    return rho_B

def wrong_trace_A(rho_AB, dA, dB):
    # Intentionally wrong: take a single A-block (a=0) instead of tracing over A.
    # This is NOT a physical reduced state and will generally change under local ops on A.
    rho_AB = rho_AB.reshape(dA, dB, dA, dB)
    return rho_AB[0, :, 0, :].copy()

def random_unitary(rng, d):
    X = rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))
    Q, R = np.linalg.qr(X)
    # make diag(R) real-positive
    ph = np.diag(R) / np.abs(np.diag(R))
    Q = Q * ph.conj()
    return Q

def singlet_state():
    # |ψ-> = (|01> - |10>)/sqrt(2)
    v = np.zeros(4, dtype=complex)
    v[1] = 1/np.sqrt(2)
    v[2] = -1/np.sqrt(2)
    rho = np.outer(v, v.conj())
    return rho

def apply_channel_A(rho_AB, kraus_ops_A, dA, dB):
    I_B = np.eye(dB, dtype=complex)
    out = np.zeros_like(rho_AB)
    for K in kraus_ops_A:
        KAB = kron(K, I_B)
        out += KAB @ rho_AB @ dagger(KAB)
    return out

def kraus_depolarizing(p):
    # depolarizing on qubit: rho -> (1-p)rho + p I/2
    # Kraus set: sqrt(1-3p/4) I, sqrt(p/4) X,Y,Z
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    a = np.sqrt(max(0.0, 1 - 3*p/4))
    b = np.sqrt(p/4)
    return [a*I, b*X, b*Y, b*Z]

def kraus_dephasing(p):
    # phase flip: rho -> (1-p)rho + p Z rho Z
    I = np.eye(2, dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    return [np.sqrt(1-p)*I, np.sqrt(p)*Z]

def kraus_amplitude_damping(gamma):
    K0 = np.array([[1,0],[0,np.sqrt(1-gamma)]], dtype=complex)
    K1 = np.array([[0,np.sqrt(gamma)],[0,0]], dtype=complex)
    return [K0, K1]

def kraus_projective_measurement_random_basis(rng):
    # Measurement on A in random basis, without postselection -> CPTP
    U = random_unitary(rng, 2)
    # projectors in computational basis rotated by U
    P0 = U @ np.array([[1,0],[0,0]], dtype=complex) @ dagger(U)
    P1 = U @ np.array([[0,0],[0,1]], dtype=complex) @ dagger(U)
    return [P0, P1]

def main():
    seed = 1
    trials = 50
    rng = np.random.default_rng(seed)

    dA = 2
    dB = 2
    rho_AB = singlet_state()

    eps = 1e-12

    print("=== T21_SV_no_signalling_tensor ===")
    print(f"NORMAL: trials={trials}, seed={seed}")

    rho_B0 = partial_trace_A(rho_AB, dA, dB)

    max_diff = 0.0

    # Random unitaries
    for _ in range(trials):
        U = random_unitary(rng, dA)
        rho2 = apply_channel_A(rho_AB, [U], dA, dB)
        rho_B2 = partial_trace_A(rho2, dA, dB)
        max_diff = max(max_diff, fro_norm(rho_B2 - rho_B0))

    # Random measurements (no postselection)
    for _ in range(trials):
        kraus = kraus_projective_measurement_random_basis(rng)
        rho2 = apply_channel_A(rho_AB, kraus, dA, dB)
        rho_B2 = partial_trace_A(rho2, dA, dB)
        max_diff = max(max_diff, fro_norm(rho_B2 - rho_B0))

    # A few standard CPTP channels
    for p in [0.3, 0.9]:
        rho2 = apply_channel_A(rho_AB, kraus_depolarizing(p), dA, dB)
        max_diff = max(max_diff, fro_norm(partial_trace_A(rho2, dA, dB) - rho_B0))
    for p in [0.5]:
        rho2 = apply_channel_A(rho_AB, kraus_dephasing(p), dA, dB)
        max_diff = max(max_diff, fro_norm(partial_trace_A(rho2, dA, dB) - rho_B0))
    for g in [0.5, 0.9]:
        rho2 = apply_channel_A(rho_AB, kraus_amplitude_damping(g), dA, dB)
        max_diff = max(max_diff, fro_norm(partial_trace_A(rho2, dA, dB) - rho_B0))

    print(f"max||Δrho_B|| = {max_diff:.3e}")
    ok = max_diff < 1e-9
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print()

    # ---------------- Negative control ----------------
    print("NEGATIVE CONTROL (intentionally WRONG trace):")
    max_diff_bad = 0.0
    rho_B0_bad = wrong_trace_A(rho_AB, dA, dB)

    # Use a mix of operations so the wrong "trace" visibly changes.
    for _ in range(trials):
        U = random_unitary(rng, dA)
        rho2 = apply_channel_A(rho_AB, [U], dA, dB)
        max_diff_bad = max(max_diff_bad, fro_norm(wrong_trace_A(rho2, dA, dB) - rho_B0_bad))

    for _ in range(trials):
        kraus = kraus_projective_measurement_random_basis(rng)
        rho2 = apply_channel_A(rho_AB, kraus, dA, dB)
        max_diff_bad = max(max_diff_bad, fro_norm(wrong_trace_A(rho2, dA, dB) - rho_B0_bad))

    for p in [0.3, 0.9]:
        rho2 = apply_channel_A(rho_AB, kraus_depolarizing(p), dA, dB)
        max_diff_bad = max(max_diff_bad, fro_norm(wrong_trace_A(rho2, dA, dB) - rho_B0_bad))

    for g in [0.5, 0.9]:
        rho2 = apply_channel_A(rho_AB, kraus_amplitude_damping(g), dA, dB)
        max_diff_bad = max(max_diff_bad, fro_norm(wrong_trace_A(rho2, dA, dB) - rho_B0_bad))

    print(f"max||Δrho_B|| = {max_diff_bad:.3e}")
    # In the negative control we WANT a big change:
    neg_ok = max_diff_bad > 1e-2
    print(f"NEGATIVE CONTROL: {'PASS' if neg_ok else 'FAIL'} (expected FAIL observed)")
    print(f"OVERALL: {'PASS' if (ok and neg_ok) else 'FAIL'}")
    return 0 if (ok and neg_ok) else 1

if __name__ == "__main__":
    raise SystemExit(main())
