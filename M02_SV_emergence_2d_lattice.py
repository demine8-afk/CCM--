#!/usr/bin/env python3
"""
M02_SV_emergence_2d_lattice.py

CCM Emergence: восстановление 2D топологии из корреляций.
Setup: 3x3 Heisenberg на торе. Dim = 2^9 = 512 — терпимо.
"""

import numpy as np
from collections import defaultdict

def pauli():
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    return I, X, Y, Z

def tensor_op(op, site, N):
    I_mat, _, _, _ = pauli()
    result = np.array([[1.0]], dtype=complex)
    for i in range(N):
        result = np.kron(result, op if i == site else I_mat)
    return result

def idx(x, y, L):
    return (y % L) * L + (x % L)

def coords(i, L):
    return i % L, i // L

def heisenberg_2d_torus(L, J=1.0):
    N = L * L
    I_mat, X, Y, Z = pauli()
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    
    for y in range(L):
        for x in range(L):
            i = idx(x, y, L)
            for dx, dy in [(1,0), (0,1)]:
                j = idx(x+dx, y+dy, L)
                for P in [X, Y, Z]:
                    H += J * tensor_op(P, i, N) @ tensor_op(P, j, N)
    return H, N

def ground_state(H):
    evals, evecs = np.linalg.eigh(H)
    return evecs[:, 0], evals[0]

def correlation_matrix(psi, N):
    _, _, _, Z = pauli()
    C = np.zeros((N, N))
    Z_ops = [tensor_op(Z, i, N) for i in range(N)]
    exp_i = [np.real(psi.conj() @ Z_ops[i] @ psi) for i in range(N)]
    
    for i in range(N):
        C[i,i] = 1.0
        for j in range(i+1, N):
            exp_ij = np.real(psi.conj() @ (Z_ops[i] @ Z_ops[j]) @ psi)
            C[i,j] = exp_ij - exp_i[i] * exp_i[j]
            C[j,i] = C[i,j]
    return C

def distance_matrix(C):
    with np.errstate(divide='ignore'):
        D = -np.log(np.clip(np.abs(C), 1e-10, 1))
    np.fill_diagonal(D, 0)
    return D

def physical_distance_torus(i, j, L):
    x1, y1 = coords(i, L)
    x2, y2 = coords(j, L)
    dx = min(abs(x1-x2), L - abs(x1-x2))
    dy = min(abs(y1-y2), L - abs(y1-y2))
    return dx + dy

def true_neighbors_torus(i, L):
    x, y = coords(i, L)
    return {idx(x+1,y,L), idx(x-1,y,L), idx(x,y+1,L), idx(x,y-1,L)}

def reconstruct_neighbors(D, k):
    N = D.shape[0]
    neighbors = {}
    for i in range(N):
        dists = [(j, D[i,j]) for j in range(N) if j != i]
        dists.sort(key=lambda x: x[1])
        neighbors[i] = set(dists[m][0] for m in range(k))
    return neighbors

def main():
    print("="*60)
    print("M02: Emergence — 2D Topology from Correlations")
    print("="*60)
    print()
    
    L = 3
    print(f"System: {L}x{L} Heisenberg torus ({L*L} spins, dim={2**(L*L)})")
    print()
    
    print("Building Hamiltonian and finding ground state...")
    H, N = heisenberg_2d_torus(L)
    psi, E0 = ground_state(H)
    print(f"Ground state energy: {E0:.4f}")
    print()
    
    print("Computing correlations...")
    C = correlation_matrix(psi, N)
    D = distance_matrix(C)
    
    # === TEST 1: Монотонность ===
    print()
    print("="*60)
    print("TEST 1: Monotonicity")
    print("="*60)
    
    groups = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            d_phys = physical_distance_torus(i, j, L)
            groups[d_phys].append(D[i,j])
    
    prev = -1
    monotonic = True
    for d in sorted(groups.keys()):
        avg = np.mean(groups[d])
        ok = avg > prev
        if not ok:
            monotonic = False
        print(f"  d_phys={d}: d_corr={avg:.4f} {'✓' if ok else '✗'}")
        prev = avg
    
    print(f"Monotonicity: {'PASS' if monotonic else 'FAIL'}")
    
    # === TEST 2: Соседи (k=4) ===
    print()
    print("="*60)
    print("TEST 2: Neighbor identification (k=4)")
    print("="*60)
    
    neighbors = reconstruct_neighbors(D, k=4)
    
    correct = 0
    for i in range(N):
        true_nn = true_neighbors_torus(i, L)
        if neighbors[i] == true_nn:
            correct += 1
        else:
            x, y = coords(i, L)
            print(f"  Site {i} ({x},{y}): expected {true_nn}, got {neighbors[i]}")
    
    nn_all = correct == N
    print(f"Correct: {correct}/{N}")
    print(f"All neighbors: {'PASS' if nn_all else 'FAIL'}")
    
    # === TEST 3: Pearson ===
    print()
    print("="*60)
    print("TEST 3: Correlation")  
    print("="*60)
    
    d_phys_list = [physical_distance_torus(i,j,L) for i in range(N) for j in range(i+1,N)]
    d_corr_list = [D[i,j] for i in range(N) for j in range(i+1,N)]
    pearson = np.corrcoef(d_phys_list, d_corr_list)[0,1]
    pearson_ok = pearson > 0.6
    print(f"Pearson: {pearson:.4f} {'PASS' if pearson_ok else 'FAIL'}")
    
    # === SUMMARY ===
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    overall = monotonic and nn_all and pearson_ok
    print(f"  Monotonicity:    {'PASS' if monotonic else 'FAIL'}")
    print(f"  Neighbors:       {'PASS' if nn_all else 'FAIL'}")
    print(f"  Pearson:         {'PASS' if pearson_ok else 'FAIL'}")
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())