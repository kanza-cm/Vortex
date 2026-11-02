#!/usr/bin/env python3
"""confronta_campi.py -- confronta due campi omega3d_matrix.bin.

Formato (identico a saetta3d.c / mpi_fissa):
    int32 nx, ny, nz  (little endian nativo x86)
    float32 dati con i piu' veloce, poi j, poi k

Se le griglie sono annidate (fattore f=2,4,... in ogni direzione) il campo
piu' fine viene ristretto con media f x f x f (come error_norm_3d.py), cosi'
si confrontano due risoluzioni diverse. Se le griglie sono identiche il
confronto e' diretto.

Uso:
    python3 scripts/confronta_campi.py out/np1/omega3d_matrix.bin out/np2/omega3d_matrix.bin
    python3 scripts/confronta_campi.py out/gpu/omega3d_matrix.bin ref/omega3d_matrix.bin --tol-rms 1e-2
    python3 scripts/confronta_campi.py a.bin b.bin --tol-rms 1e-5 --label "np1" --label "np2"

Metriche: ||err||_inf, rms(err), rms(err)/rms(riferimento).
Codici di uscita: 0 = entro tolleranza, 1 = fuori, 2 = errore input.
Se numpy e' disponibile il calcolo e' veloce; altrimenti c'e' un fallback puro
Python (molto piu' lento: usarlo solo per griglie piccole).
"""
import sys
import struct
import argparse


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 12:
        raise ValueError("file troppo corto")
    nx, ny, nz = struct.unpack("iii", raw[:12])
    n = nx * ny * nz
    if len(raw) < 12 + 4 * n:
        raise ValueError("dati incompleti: attesi %d float, trovati %d"
                         % (n, (len(raw) - 12) // 4))
    return nx, ny, nz, raw[12:12 + 4 * n]


def compare_numpy(pa, pb):
    import numpy as np
    na = np.fromfile(pa, dtype=np.int32, count=3)
    fa = np.fromfile(pa, dtype=np.float32, offset=12)
    nb = np.fromfile(pb, dtype=np.int32, count=3)
    fb = np.fromfile(pb, dtype=np.float32, offset=12)
    dims_a = tuple(int(x) for x in na)
    dims_b = tuple(int(x) for x in nb)
    va = fa.reshape(dims_a[::-1])   # (nz, ny, nx)
    vb = fb.reshape(dims_b[::-1])
    # restrizione del piu' fine sul piu' grossolano
    if dims_a != dims_b:
        f = [dims_b[i] // dims_a[i] for i in range(3)]
        if any(dims_b[i] % dims_a[i] for i in range(3)) or len(set(f)) != 1:
            raise ValueError("griglie non annidate: %s vs %s" % (dims_a, dims_b))
        f = f[0]
        if f < 1:
            raise ValueError("griglie non annidate (a piu' fine di b)")
        if f > 1:
            vb = vb.reshape(dims_a[0], f, dims_a[1], f, dims_a[2], f).mean(axis=(1, 3, 5))
    d = va.astype(np.float64) - vb.astype(np.float64)
    linf = float(np.max(np.abs(d)))
    rms = float(np.sqrt(np.mean(d * d)))
    ref = float(np.sqrt(np.mean(vb.astype(np.float64) ** 2)))
    return linf, rms, (rms / ref if ref > 0 else 0.0), dims_a


def compare_pure(pa, pb):
    nxa, nya, nza, ra = load(pa)
    nxb, nyb, nzb, rb = load(pb)
    va = memoryview(ra).cast("f")
    vb = memoryview(rb).cast("f")
    if (nxa, nya, nza) != (nxb, nyb, nzb):
        f = nxb // nxa
        if f < 1 or nxb % nxa or nyb % nya or nzb % nza:
            raise ValueError("griglie non annidate")
        # restrizione pura python (lenta)
        nx, ny, nz = nxa, nya, nza
        fine = vb
        for _ in range(int(round(f.bit_length() - 1))):
            NX, NY, NZ = nx // 2, ny // 2, nz // 2
            coarse = [0.0] * (NX * NY * NZ)
            inv = 1.0 / 8.0
            for K in range(NZ):
                for J in range(NY):
                    for I in range(NX):
                        s = 0.0
                        for ck in range(2):
                            kb = (K * 2 + ck) * ny + J * 2
                            for cj in range(2):
                                base = (kb + cj) * nx + I * 2
                                for ci in range(2):
                                    s += fine[base + ci]
                        coarse[(K * NY + J) * NX + I] = s * inv
            nx, ny, nz = NX, NY, NZ
            fine = coarse
        vb = fine
    s2 = 0.0
    linf = 0.0
    n = nxa * nya * nza
    for i in range(n):
        d = va[i] - vb[i]
        if d < 0:
            d = -d
        if d > linf:
            linf = d
        s2 += d * d
    rms = (s2 / n) ** 0.5
    ref = 0.0
    s2r = 0.0
    for i in range(n):
        s2r += vb[i] * vb[i]
    ref = (s2r / n) ** 0.5
    return linf, rms, (rms / ref if ref > 0 else 0.0), (nxa, nya, nza)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--tol-rms", type=float, default=1e-2,
                    help="tolleranza su rms(err)/rms(riferimento) (default 1e-2)")
    ap.add_argument("--label", action="append", default=[])
    args = ap.parse_args()
    la = args.label[0] if len(args.label) > 0 else "A"
    lb = args.label[1] if len(args.label) > 1 else "B"
    try:
        try:
            linf, rms, rel, dims = compare_numpy(args.file_a, args.file_b)
        except ImportError:
            linf, rms, rel, dims = compare_pure(args.file_a, args.file_b)
    except Exception as e:  # noqa: BLE001
        print("ERRORE:", e, file=sys.stderr)
        return 2
    print("# confronto %s (%s) vs %s (%s)" % (args.file_a, la, args.file_b, lb))
    print("# griglia comune: %d x %d x %d" % dims)
    print("linf        = %.6e" % linf)
    print("rms_err     = %.6e" % rms)
    print("rms_rel     = %.6e   (tolleranza %.3e)" % (rel, args.tol_rms))
    if rel <= args.tol_rms:
        print("ESITO: entro tolleranza")
        return 0
    print("ESITO: FUORI TOLLERANZA")
    return 1


if __name__ == "__main__":
    sys.exit(main())
