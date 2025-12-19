#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T19_SV_multiplicativity

Single-file, single-run test.

Claim
-----
For independent subsystems (or independent history pieces), action additivity implies
amplitude multiplicativity:
  S_joint = S_A + S_B
  A_joint = exp(i S_joint) = exp(i S_A) * exp(i S_B)

Includes a negative control: inject an extra phase delta into the "joint" amplitude,
which must break multiplicativity by ~delta.

No external deps beyond numpy.
"""

import numpy as np

def main():
    seed = 1
    trials = 2000
    rng = np.random.default_rng(seed)

    print("=== T19_SV_multiplicativity ===")
    print(f"NORMAL: trials={trials}, seed={seed}")

    max_err = 0.0
    for _ in range(trials):
        SA = rng.uniform(-10.0, 10.0)
        SB = rng.uniform(-10.0, 10.0)
        A_joint = np.exp(1j*(SA + SB))
        A_prod  = np.exp(1j*SA) * np.exp(1j*SB)
        max_err = max(max_err, float(np.abs(A_joint - A_prod)))

    print(f"max|A_joint - A_A*A_B| = {max_err:.3e}")
    ok = max_err < 1e-9
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print()

    print("NEGATIVE CONTROL (extra phase injected):")
    delta = 1.0e-3
    max_err_bad = 0.0
    for _ in range(trials):
        SA = rng.uniform(-10.0, 10.0)
        SB = rng.uniform(-10.0, 10.0)
        A_joint_bad = np.exp(1j*(SA + SB + delta))
        A_prod      = np.exp(1j*SA) * np.exp(1j*SB)
        max_err_bad = max(max_err_bad, float(np.abs(A_joint_bad - A_prod)))

    print(f"delta={delta:.1e}")
    print(f"max|A_joint - A_A*A_B| = {max_err_bad:.3e}")
    neg_ok = max_err_bad > 1e-6
    print(f"NEGATIVE CONTROL: {'PASS' if neg_ok else 'FAIL'} (expected FAIL observed)")
    print(f"OVERALL: {'PASS' if (ok and neg_ok) else 'FAIL'}")
    return 0 if (ok and neg_ok) else 1

if __name__ == "__main__":
    raise SystemExit(main())
