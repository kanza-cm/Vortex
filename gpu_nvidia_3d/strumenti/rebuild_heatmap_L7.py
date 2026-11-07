#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ricalcola la riga L7 delle heatmap AMR (T=30) con le run corrette
(ref_ratio 4 2) e rigenera i png, per entrambe le configurazioni."""
import os, sys, glob, math, subprocess
BASE = "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d"
AMR  = "/home/zorka/simulazioni/Test 1/gpu_amr_3d"
ENORM = os.path.join(AMR, "strumenti", "error_norm_amr3d.py")
THRS = ["0.3", "0.1", "3e-2"]
CASES = [("vortici_inclinati", "vortici_inclinati"), ("vortici_non_inclinati", "vortici_non_inclinati")]

def last_pf(d):
    pfs = [p for p in glob.glob(os.path.join(d, "plt*")) if os.path.isdir(p)]
    if not pfs:
        sys.exit("nessun plotfile in " + d)
    return max(pfs, key=os.path.getmtime)

def err(pf, ref):
    out = subprocess.check_output([sys.executable, ENORM, pf, ref, "--minlevel", "4"], text=True).strip()
    v = out.split()
    return float(v[0]), float(v[1]), int(v[2])

for case, _ in CASES:
    d = os.path.join(BASE, case, "heatmap_adattiva")
    ref = last_pf(os.path.join(d, "L8_thr3e-2"))
    new = {}
    for i, thr in enumerate(THRS, start=1):
        rd = os.path.join(BASE, case, "heatmap_adattiva_fix", "L7_thr" + thr)
        pf = last_pf(rd)
        ei, e2, cells = err(pf, ref)
        new[i] = (ei, e2, cells)
        print(f"{case} thr={thr}: linf={ei:.3e} l2={e2:.3e} cells={cells}")
    for fname, idx in (("heatmap.dat", 1), ("heatmap_l2.dat", 2)):
        p = os.path.join(d, fname)
        lines = open(p).read().splitlines()
        out = []
        for ln in lines:
            if ln.startswith("#") or not ln.strip():
                out.append(ln); continue
            v = ln.split()
            if int(v[1]) == 7:
                j = int(v[0])
                e = new[j][0] if idx == 1 else new[j][1]
                cells = new[j][2]
                out.append(f"{j} 7 {math.log10(e):.2f} {cells} 0")
            else:
                out.append(ln)
        open(p, "w").write("\n".join(out) + "\n")
        print("aggiornato", p)
    # rigenera entrambi i png con i gp esistenti
    for gp in ("heatmap.gp", "heatmap_l2.gp"):
        subprocess.check_call(["gnuplot", gp], cwd=d)
    print("png rigenerati per", case)
