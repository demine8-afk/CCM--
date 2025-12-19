#!/usr/bin/env python3
"""
M01_SV_emergence_correlation_metric.py

CCM Emergence: 1D топология из корреляций + negative control.
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

def heisenberg_ring(N, J=1.0):
    I_mat, X, Y, Z = pauli()
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        for P in [X, Y, Z]:
            H += J * tensor_op(P, i, N) @ tensor_op(P, j, N)
    return H

def ground_state(H):
    evals, evecs = np.linalg.eigh(H)
    return evecs[:, 0]

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

def physical_distance_ring(i, j, N):
    diff = abs(i - j)
    return min(diff, N - diff)

def reconstruct_graph(D, k=2):
    N = D.shape[0]
    edges = set()
    neighbors = {}
    for i in range(N):
        dists = [(j, D[i,j]) for j in range(N) if j != i]
        dists.sort(key=lambda x: x[1])
        nearest = [dists[m][0] for m in range(k)]
        neighbors[i] = set(nearest)
        for j in nearest:
            edges.add(tuple(sorted([i, j])))
    return edges, neighbors

def main():
    print("="*60)
    print("M01: Emergence — 1D Topology from Correlations")
    print("="*60)
    
    N = 8
    
    # === TEST A: Physical system ===
    print(f"\nTEST A: Physical system ({N}-spin Heisenberg ring)")
    print("-"*50)
    
    H = heisenberg_ring(N)
    psi = ground_state(H)
    C = correlation_matrix(psi, N)
    D = distance_matrix(C)
    
    # Монотонность
    groups = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            groups[physical_distance_ring(i,j,N)].append(D[i,j])
    
    prev, monotonic = -1, True
    for d in sorted(groups.keys()):
        avg = np.mean(groups[d])
        if avg <= prev: monotonic = False
        prev = avg
    print(f"Monotonicity: {'PASS' if monotonic else 'FAIL'}")
    
    # Топология
    edges, neighbors = reconstruct_graph(D, k=2)
    true_edges = {tuple(sorted([i, (i+1)%N])) for i in range(N)}
    edges_match = edges == true_edges
    print(f"Topology recovered: {'PASS' if edges_match else 'FAIL'}")
    
    # Соседи
    nn_ok = all(neighbors[i] == {(i-1)%N, (i+1)%N} for i in range(N))
    print(f"Neighbors correct: {'PASS' if nn_ok else 'FAIL'}")
    
    # Pearson
    pairs = [(physical_distance_ring(i,j,N), D[i,j]) 
             for i in range(N) for j in range(i+1,N)]
    pearson = np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0,1]
    pearson_ok = pearson > 0.8
    print(f"Pearson: {pearson:.4f} {'PASS' if pearson_ok else 'FAIL'}")
    
    phys_pass = monotonic and edges_match and nn_ok and pearson_ok
    
    # === TEST B: Negative control ===
    print(f"\nTEST B: Negative control (random correlations)")
    print("-"*50)
    
    rng = np.random.default_rng(42)
    C_rand = rng.uniform(-1, 1, (N, N))
    C_rand = (C_rand + C_rand.T) / 2
    np.fill_diagonal(C_rand, 1.0)
    D_rand = distance_matrix(C_rand)
    
    edges_rand, _ = reconstruct_graph(D_rand, k=2)
    rand_match = edges_rand == true_edges
    nc_pass = not rand_match
    print(f"Random gives true topology: {rand_match}")
    print(f"Negative control: {'PASS' if nc_pass else 'FAIL'}")
    
    # === SUMMARY ===
    print("\n" + "="*60)
    overall = phys_pass and nc_pass
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    
    return 0 if overall else 1

if __name__ == "__main__":
    raise SystemExit(main())