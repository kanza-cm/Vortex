#!/usr/bin/env python3
"""Infinity norm (and L2) of the error between two fields saved by
Basilisk output_matrix(), on nested uniform grids.

Usage:  error_norm.py coarse.bin reference.bin

The reference (finer) field is averaged in 2x2 blocks until it matches
the coarse grid (restriction, as in Basilisk), then we compute:
    ||err||_inf = max |coarse - restricted_reference|

output_matrix format:
    float ny ; float y[0..ny-1] ; for each x: float x, float z[0..ny-1]
"""
import sys
from array import array

HUGE = 1e29  # nodata in Basilisk


def load(fn):
    """Return (nx, ny, values) with values indexed i*ny + j."""
    a = array('f')
    with open(fn, 'rb') as f:
        a.frombytes(f.read())
    if len(a) < 1:
        return 0, 0, array('f')
    ny = int(a[0])
    start = 1 + ny
    nx = (len(a) - start) // (1 + ny)
    vals = array('f')
    pos = start
    for _ in range(nx):
        pos += 1                       # skip x coordinate
        vals.extend(a[pos:pos + ny])
        pos += ny
    return nx, ny, vals


def block_average(vals, nx, ny, f):
    """Block average f x f -> new grid (nx/f) x (ny/f)."""
    if f == 1:
        return nx, ny, vals
    NX, NY = nx // f, ny // f
    out = array('f')
    inv = 1.0 / (f * f)
    for I in range(NX):
        for J in range(NY):
            s = 0.0
            for ci in range(f):
                base = (I * f + ci) * ny + J * f
                for cj in range(f):
                    s += vals[base + cj]
            out.append(s * inv)
    return NX, NY, out


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("uso: error_norm.py coarse.bin reference.bin\n")
        return 1
    nxc, nyc, vc = load(sys.argv[1])
    nxr, nyr, vr = load(sys.argv[2])
    if nxc == 0 or nxr == 0:
        sys.stderr.write("file vuoto\n")
        return 1
    if nxr % nxc or nyr % nyc:
        sys.stderr.write("griglie non annidate (%dx%d vs %dx%d)\n"
                         % (nxc, nyc, nxr, nyr))
        return 1
    f = nxr // nxc
    NX, NY, vref = block_average(vr, nxr, nyr, f)
    if NX != nxc or NY != nyc:
        sys.stderr.write("dimensioni diverse dopo la media\n")
        return 1

    linf = 0.0
    s2 = 0.0
    n = 0
    for x, y in zip(vc, vref):
        if abs(x) > HUGE or abs(y) > HUGE:
            continue
        d = x - y
        ad = -d if d < 0 else d
        if ad > linf:
            linf = ad
        s2 += d * d
        n += 1
    l2 = (s2 / n) ** 0.5 if n else 0.0
    print("%.6e %.6e" % (linf, l2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
