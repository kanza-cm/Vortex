#!/usr/bin/env python3
"""L_inf and L2 error between two 3D AMR runs (incflo plotfiles) on a common
coarse grid.

Both fields are volume-averaged down to level `minlevel` of the coarser run
(MINLEVEL), then compared cell by cell.

Usage:
    error_norm_amr3d.py <plotfile_a> <plotfile_b> --minlevel 4 [--field vort]

Prints one line: "<linf> <l2> <cells_a> <cells_b>" where cells_* is the
number of cells of the whole hierarchy in each run."""
import os, sys, argparse
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_amr_plotfile import read_header, read_offsets, read_fab_comp

NAN = float("nan")


def _dims(H, lev):
    dx = H["levels"][lev]["dx"]
    problo, probhi = H["problo"], H["probhi"]
    return tuple(int(round((probhi[d] - problo[d]) / dx[d])) for d in range(3))


def _idx(H, L, blo, bhi):
    problo = H["problo"]
    dx = L["dx"]
    ilo = [int(round((blo[d] - problo[d]) / dx[d])) for d in range(3)]
    ihi = [int(round((bhi[d] - problo[d]) / dx[d])) - 1 for d in range(3)]
    return ilo, ihi


def load_field(pf, field, minlevel):
    H = read_header(pf)
    if field not in H["varnames"]:
        sys.exit("field '%s' not in %s (have %s)" % (field, pf, H["varnames"]))
    icomp = H["varnames"].index(field)
    problo = H["problo"]
    levels = [(L, _dims(H, L["lev"])) for L in H["levels"]]
    maxlev = len(levels) - 1
    n0 = levels[0][1]      # common grid = base level of the runs (2^MINLEVEL)
    if n0[0] != (1 << minlevel):
        sys.stderr.write("attenzione: livello 0 = %d celle, atteso 2^%d\n" % (n0[0], minlevel))

    # coverage[lev]: level-lev cells covered by a finer level (skipped there)
    coverage = [None] * (maxlev + 1)
    for lev in range(1, maxlev + 1):
        Lp, dims_p = levels[lev - 1]
        Lc, dims_c = levels[lev]
        rat = tuple(dims_c[d] // dims_p[d] for d in range(3))
        cov = bytearray(dims_p[0] * dims_p[1] * dims_p[2])
        for (blo, bhi) in Lc["boxes"]:
            ilo, ihi = _idx(H, Lc, blo, bhi)
            plo = [ilo[d] // rat[d] for d in range(3)]
            phi = [ihi[d] // rat[d] for d in range(3)]
            for k in range(plo[2], phi[2] + 1):
                for j in range(plo[1], phi[1] + 1):
                    base = (k * dims_p[1] + j) * dims_p[0]
                    for i in range(plo[0], phi[0] + 1):
                        cov[base + i] = 1
        coverage[lev - 1] = cov

    num = [0.0] * (n0[0] * n0[1] * n0[2])
    den = [0.0] * (n0[0] * n0[1] * n0[2])
    cells = 0

    for lev, (L, dims) in enumerate(levels):
        factor = tuple(dims[d] // n0[d] for d in range(3))
        w = 1.0 / (factor[0] * factor[1] * factor[2])
        cov = coverage[lev] if lev < maxlev else None
        data = array("f", [NAN]) * (dims[0] * dims[1] * dims[2])
        ddir = os.path.dirname(L["path"])
        offs = read_offsets(os.path.join(ddir, "Cell_H"))
        for (fn, off), (blo, bhi) in zip(offs, L["boxes"]):
            ilo, ihi = _idx(H, L, blo, bhi)
            nx = ihi[0] - ilo[0] + 1
            ny = ihi[1] - ilo[1] + 1
            nz = ihi[2] - ilo[2] + 1
            fab = read_fab_comp(os.path.join(ddir, fn), off, icomp, nx, ny, nz)
            cells += nx * ny * nz
            for k in range(nz):
                for j in range(ny):
                    bf = (k * ny + j) * nx
                    bg = ((ilo[2] + k) * dims[1] + (ilo[1] + j)) * dims[0] + ilo[0]
                    for i in range(nx):
                        data[bg + i] = fab[bf + i]
        for k in range(dims[2]):
            for j in range(dims[1]):
                base = (k * dims[1] + j) * dims[0]
                for i in range(dims[0]):
                    v = data[base + i]
                    if v != v:          # NaN: not covered by this level
                        continue
                    if cov is not None and cov[base + i]:
                        continue        # a finer level takes care of it
                    t = ((k // factor[2]) * n0[1] + (j // factor[1])) * n0[0] + (i // factor[0])
                    num[t] += v * w
                    den[t] += w

    coarse = array("f", [NAN]) * (n0[0] * n0[1] * n0[2])
    for i in range(len(coarse)):
        if den[i] > 0.0:
            coarse[i] = num[i] / den[i]
    return coarse, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plotfile_a")
    ap.add_argument("plotfile_b")
    ap.add_argument("--minlevel", type=int, default=4)
    ap.add_argument("--field", default="vort")
    a = ap.parse_args()
    fa, ca = load_field(a.plotfile_a, a.field, a.minlevel)
    fb, cb = load_field(a.plotfile_b, a.field, a.minlevel)
    linf = 0.0
    s2 = 0.0
    n = 0
    for x, y in zip(fa, fb):
        if x != x or y != y:
            continue
        d = x - y
        ad = -d if d < 0 else d
        if ad > linf:
            linf = ad
        s2 += d * d
        n += 1
    l2 = (s2 / n) ** 0.5 if n else 0.0
    print("%.6e %.6e %d %d" % (linf, l2, ca, cb))


if __name__ == "__main__":
    main()
