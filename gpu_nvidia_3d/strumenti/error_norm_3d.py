#!/usr/bin/env python3
"""Infinity norm (and L2) of the error between two omega3d_matrix.bin fields
saved by saetta3d.c (matrix event) on nested regular 3D grids.

Usage:  error_norm_3d.py coarse.bin reference.bin

The reference (finer) is averaged in f x f x f blocks (f = 2 for
each level of difference) until it matches the coarse grid
(restriction, as in Basilisk), then it computes:

    ||err||_inf = max |coarse - riferimento_ristretto|
    ||err||_2   = rms  (coarse - riferimento_ristretto)

Like error_norm.py (2D) it is deliberately pure python (no numpy):
the dominant cost is the restriction loop (O(cells of the reference)).

Format of omega3d_matrix.bin:
    int32 nx, ny, nz ; then float32 with i fastest, then j, then k.
"""
import sys
import struct
from array import array


def load(fn):
    """Returns (nx, ny, nz, values) indexed (k*ny + j)*nx + i."""
    with open(fn, 'rb') as f:
        raw = f.read()
    nx, ny, nz = struct.unpack('iii', raw[:12])
    v = array('f')
    v.frombytes(raw[12:12 + nx * ny * nz * 4])
    return nx, ny, nz, v


def restrict(v, nx, ny, nz, f):
    """Averages in f x f x f blocks -> grid (nx/f, ny/f, nz/f)."""
    if f == 1:
        return nx, ny, nz, v
    NX, NY, NZ = nx // f, ny // f, nz // f
    out = array('f')
    out.frombytes(bytes(4 * NX * NY * NZ))
    inv = 1.0 / (f * f * f)
    for K in range(NZ):
        for J in range(NY):
            for I in range(NX):
                s = 0.0
                for ck in range(f):
                    kb = (K * f + ck) * ny + J * f
                    for cj in range(f):
                        base = (kb + cj) * nx + I * f
                        for ci in range(f):
                            s += v[base + ci]
                out[(K * NY + J) * NX + I] = s * inv
    return NX, NY, NZ, out


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("uso: error_norm_3d.py coarse.bin reference.bin\n")
        return 1
    nxc, nyc, nzc, vc = load(sys.argv[1])
    nxr, nyr, nzr, vr = load(sys.argv[2])

    f = nxr // nxc if nxc else 0
    if (nxc == 0 or nxr == 0 or f < 1 or
            nxr % nxc or nyr % nyc or nzr % nzc or
            nyr // nyc != f or nzr // nzc != f):
        sys.stderr.write("griglie non annidate (%dx%dx%d vs %dx%dx%d)\n"
                         % (nxc, nyc, nzc, nxr, nyr, nzr))
        return 1

    NX, NY, NZ, vref = restrict(vr, nxr, nyr, nzr, f)
    if len(vc) != len(vref):
        sys.stderr.write("dimensioni diverse dopo la restrizione\n")
        return 1

    linf = 0.0
    s2 = 0.0
    n = 0
    for a, b in zip(vc, vref):
        d = a - b
        if d < 0:
            d = -d
        if d > linf:
            linf = d
        s2 += d * d
        n += 1
    l2 = (s2 / n) ** 0.5 if n else 0.0
    print("%.6e %.6e" % (linf, l2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
