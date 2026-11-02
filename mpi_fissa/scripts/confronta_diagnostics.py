#!/usr/bin/env python3
"""confronta_diagnostics.py -- confronta due file diagnostics.dat.

I file hanno lo stesso formato (saetta3d.c / mpi_fissa):
    # t dt it enstrophy energy umax wmax circulation_slice separation xc yc cells ...
      t dt it enstrophy energy umax wmax circulation_slice separation xc yc cells ...

Uso tipico:
    # prototipo (1 rank) contro riferimento saetta3d
    python3 scripts/confronta_diagnostics.py \
        out/cpu_L6_np1/diagnostics.dat \
        gpu_nvidia_3d/risultati/.../diagnostics.dat \
        --label "prototipo np1" --label "saetta3d L6"

    # prototipo np1 contro prototipo np2 (test di invarianza MPI)
    python3 scripts/confronta_diagnostics.py out/np1/diagnostics.dat out/np2/diagnostics.dat \
        --tiny

Colonne confrontate: enstrophy, energy, umax, wmax, circulation_slice, separation
(colonne 4,5,6,7,8,9 degli indici a base 0 -> 3..8).
Le righe sono accoppiate per tempo entro --toll-t (default 1e-4).
Codici di uscita: 0 = entro tolleranza, 1 = fuori tolleranza, 2 = errore input.
"""
import sys
import argparse

COLS = [("enstrophy", 3), ("energy", 4), ("umax", 5), ("wmax", 6),
        ("G_slice", 7), ("separation", 8)]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) < 10:
                continue
            rows.append(vals)
    rows.sort(key=lambda r: r[0])
    return rows


def match_rows(a, b, toll_t):
    """accoppia le righe per tempo (a e b ordinate per tempo)"""
    import bisect
    times = [r[0] for r in b]
    out = []
    for ra in a:
        i = bisect.bisect_left(times, ra[0])
        best = None
        bestd = 1e30
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(b):
                d = abs(b[j][0] - ra[0])
                if d < bestd:
                    bestd, best = d, j
        if best is not None and bestd <= toll_t:
            out.append((ra, b[best]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label", action="append", default=[],
                    help="etichetta per i due file (due volte)")
    ap.add_argument("--toll-t", type=float, default=1e-4)
    ap.add_argument("--tiny", action="store_true",
                    help="tolleranze da invarianza MPI (1e-5) invece che di fisica")
    ap.add_argument("--tol-strict", action="store_true", help="tolleranze strette")
    args = ap.parse_args()

    la = args.label[0] if len(args.label) > 0 else "A"
    lb = args.label[1] if len(args.label) > 1 else "B"
    a = load(args.file_a)
    b = load(args.file_b)
    if not a or not b:
        print("ERRORE: file vuoto o illeggibile", file=sys.stderr)
        return 2
    pairs = match_rows(a, b, args.toll_t)
    if not pairs:
        print("ERRORE: nessun tempo in comune (file con tempi diversi?)", file=sys.stderr)
        return 2

    if args.tiny:
        tol = {"enstrophy": 1e-4, "energy": 1e-4, "umax": 1e-4,
               "wmax": 1e-4, "G_slice": 1e-4, "separation": 1e-4}
    elif args.tol_strict:
        tol = {"enstrophy": 1e-3, "energy": 1e-3, "umax": 1e-3,
               "wmax": 3e-3, "G_slice": 1e-3, "separation": 3e-3}
    else:
        tol = {"enstrophy": 5e-2, "energy": 5e-2, "umax": 1e-1,
               "wmax": 1e-1, "G_slice": 3e-2, "separation": 5e-2}

    print("# confronto: A=%s  B=%s" % (la, lb))
    print("# tempi in comune: %d (da t=%.6g a t=%.6g)" % (len(pairs), pairs[0][0][0], pairs[-1][0][0]))
    print()
    maxrel = {k: 0.0 for k, _ in COLS}
    for ra, rb in pairs:
        for name, col in COLS:
            va, vb = ra[col], rb[col]
            den = max(abs(vb), 1e-30)
            rel = abs(va - vb) / den
            if rel > maxrel[name]:
                maxrel[name] = rel

    print("%-14s %12s %12s %10s" % ("quantita", "max_rel", "tolleranza", "esito"))
    fail = False
    for name, _ in COLS:
        esito = "OK" if maxrel[name] <= tol[name] else "FUORI"
        if esito == "FUORI":
            fail = True
        print("%-14s %12.3e %12.3e %10s" % (name, maxrel[name], tol[name], esito))
    # stampa dei punti campione
    print()
    print("punti campione (t, %s, %s):" % (la, lb))
    for ra, rb in pairs[:12]:
        t = ra[0]
        print("  t=%-8g  Z: %.6e / %.6e   sep: %.6e / %.6e"
              % (t, ra[3], rb[3], ra[8], rb[8]))
    if len(pairs) > 12:
        print("  ... (%d tempi in comune)" % len(pairs))
    print()
    if fail:
        print("ESITO: FUORI TOLLERANZA")
        return 1
    print("ESITO: entro tolleranza")
    return 0


if __name__ == "__main__":
    sys.exit(main())
