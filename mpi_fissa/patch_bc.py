#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch.py -- allineamento delle BC/viscosita' di mpi_fissa.cu a saetta3d.c.

Uso (dalla cartella di questo file, o con --file):
    python3 patch.py --check                  # sola verifica (NON scrive nulla)
    python3 patch.py --apply                  # applica + backup .bak-<timestamp>
    python3 patch.py --out /tmp/mpi_fissa_patched.cu   # scrive una copia patchata
    python3 patch.py --apply --with-dt-policy # include anche la politica del dt

Il file di default e' il sorgente del prototipo nel workspace WSL.
Nessun comando tocca il workspace se non con --apply (che fa il backup prima).

Le patch sono SOSTITUZIONI DI STRINGHE ESATTE, verificate con conteggio
occorrenze (deve essere esattamente 1). Se il conteggio non e' 1 lo script
esce con codice 2 senza scrivere niente.
"""

import argparse
import datetime
import os
import shutil
import sys

DEFAULT_FILE = os.path.join(
    "\\\\wsl.localhost", "Ubuntu", "home", "zorka", "simulazioni", "Test 1",
    "mpi_fissa", "src", "mpi_fissa.cu")
# fallback: path POSIX (se eseguito dentro WSL)
FALLBACK_FILE = "/home/zorka/simulazioni/Test 1/mpi_fissa/src/mpi_fissa.cu"


# --------------------------------------------------------------------------
# PATCH 1 -- operatore viscoso: div(2 mu D) invece del laplaciano semplice
# --------------------------------------------------------------------------
# Il riferimento (basilisk/src/viscosity.h, ramo non-TREE usato dalla griglia
# fissa) usa l'operatore completo div(mu (grad u + grad u^T)), che sul campo
# centrato differisce dal laplaciano per il termine mu*grad(div u). Alle pareti
# la differenza non e' piccola: la componente normale e' dispari rispetto alla
# parete, quindi la derivata normale pesa il DOPPIO.
# Footprint +/-1: usa solo i ghost che bc_u() gia' riempie.

OLD_VISC = '''        float lapx = (u.x[IDX(g, k, j, i + 1)] + u.x[IDX(g, k, j, i - 1)]
                    + u.x[IDX(g, k, j + 1, i)] + u.x[IDX(g, k, j - 1, i)]
                    + u.x[IDX(g, k + 1, j, i)] + u.x[IDX(g, k - 1, j, i)] - 6.f*u.x[p])/dh2;
        float lapy = (u.y[IDX(g, k, j, i + 1)] + u.y[IDX(g, k, j, i - 1)]
                    + u.y[IDX(g, k, j + 1, i)] + u.y[IDX(g, k, j - 1, i)]
                    + u.y[IDX(g, k + 1, j, i)] + u.y[IDX(g, k - 1, j, i)] - 6.f*u.y[p])/dh2;
        float lapz = (u.z[IDX(g, k, j, i + 1)] + u.z[IDX(g, k, j, i - 1)]
                    + u.z[IDX(g, k, j + 1, i)] + u.z[IDX(g, k, j - 1, i)]
                    + u.z[IDX(g, k + 1, j, i)] + u.z[IDX(g, k - 1, j, i)] - 6.f*u.z[p])/dh2;
        u.x[p] += c*lapx;
        u.y[p] += c*lapy;
        u.z[p] += c*lapz;'''

NEW_VISC = '''        /* div(2 D) e non lap(u): stessa discretizzazione di
         * basilisk/src/viscosity.h (ramo non-TREE) --
         *   x: 2*d2xx + d2yy + d2zz + d2(u.y)/dxdy + d2(u.z)/dxdz  ecc.
         * Il fattore 2 sulla direzione della componente e' il termine
         * grad(div u) che il laplaciano semplice non ha; alle pareti vale
         * esattamente il doppio della derivata normale (componente normale
         * dispari rispetto alla parete). Footprint +/-1: ghost gia' presenti. */
        float ux = u.x[p], uy = u.y[p], uz = u.z[p];
        float d2xx_x = u.x[IDX(g, k, j, i + 1)] - 2.f*ux + u.x[IDX(g, k, j, i - 1)];
        float d2yy_x = u.x[IDX(g, k, j + 1, i)] - 2.f*ux + u.x[IDX(g, k, j - 1, i)];
        float d2zz_x = u.x[IDX(g, k + 1, j, i)] - 2.f*ux + u.x[IDX(g, k - 1, j, i)];
        float d2xx_y = u.y[IDX(g, k, j, i + 1)] - 2.f*uy + u.y[IDX(g, k, j, i - 1)];
        float d2yy_y = u.y[IDX(g, k, j + 1, i)] - 2.f*uy + u.y[IDX(g, k, j - 1, i)];
        float d2zz_y = u.y[IDX(g, k + 1, j, i)] - 2.f*uy + u.y[IDX(g, k - 1, j, i)];
        float d2xx_z = u.z[IDX(g, k, j, i + 1)] - 2.f*uz + u.z[IDX(g, k, j, i - 1)];
        float d2yy_z = u.z[IDX(g, k, j + 1, i)] - 2.f*uz + u.z[IDX(g, k, j - 1, i)];
        float d2zz_z = u.z[IDX(g, k + 1, j, i)] - 2.f*uz + u.z[IDX(g, k - 1, j, i)];
        /* trasposte: 0.25*(f(+,+) - f(-,+) - f(+,-) + f(-,-)) sui 4 angoli */
        float cxy = 0.25f*(u.y[IDX(g, k, j + 1, i + 1)] - u.y[IDX(g, k, j + 1, i - 1)]
                         - u.y[IDX(g, k, j - 1, i + 1)] + u.y[IDX(g, k, j - 1, i - 1)]);
        float cxz = 0.25f*(u.z[IDX(g, k + 1, j, i + 1)] - u.z[IDX(g, k + 1, j, i - 1)]
                         - u.z[IDX(g, k - 1, j, i + 1)] + u.z[IDX(g, k - 1, j, i - 1)]);
        float cyx = 0.25f*(u.x[IDX(g, k, j + 1, i + 1)] - u.x[IDX(g, k, j - 1, i + 1)]
                         - u.x[IDX(g, k, j + 1, i - 1)] + u.x[IDX(g, k, j - 1, i - 1)]);
        float cyz = 0.25f*(u.z[IDX(g, k + 1, j + 1, i)] - u.z[IDX(g, k + 1, j - 1, i)]
                         - u.z[IDX(g, k - 1, j + 1, i)] + u.z[IDX(g, k - 1, j - 1, i)]);
        float czx = 0.25f*(u.x[IDX(g, k + 1, j, i + 1)] - u.x[IDX(g, k - 1, j, i + 1)]
                         - u.x[IDX(g, k + 1, j, i - 1)] + u.x[IDX(g, k - 1, j, i - 1)]);
        float czy = 0.25f*(u.y[IDX(g, k + 1, j + 1, i)] - u.y[IDX(g, k - 1, j + 1, i)]
                         - u.y[IDX(g, k + 1, j - 1, i)] + u.y[IDX(g, k - 1, j - 1, i)]);
        float lx = 2.f*d2xx_x + d2yy_x + d2zz_x + cxy + cxz;
        float ly = 2.f*d2yy_y + d2xx_y + d2zz_y + cyx + cyz;
        float lz = 2.f*d2zz_z + d2xx_z + d2yy_z + czx + czy;
        u.x[p] += c*lx/dh2;
        u.y[p] += c*ly/dh2;
        u.z[p] += c*lz/dh2;'''

# --------------------------------------------------------------------------
# PATCH 2 (opzionale) -- politica del passo come basilisk/src/timestep.h
# --------------------------------------------------------------------------
# timestep() di Basilisk limita la crescita con
#     if (dt > previous) dt = (previous + 0.1*dt)/1.1
# e non con dt <= 1.1*dt_prev. Effetto: il riferimento parte da ~0.09*dt_CFL e
# si avvicina al limite CFL solo dopo ~30 passi -> a parita' di CFL nominale fa
# piu' passi (52 contro 41 a CFL=0.1 su NX=64). Serve per confrontare gli stessi
# dt, non e' una modifica alle BC.

OLD_DT = '''    if (dt > 1.1*dt_prev) dt = 1.1*dt_prev;   /* crescita max 10%/passo */'''

NEW_DT = '''    /* politica di basilisk/src/timestep.h: la crescita e' limitata da
     * (previous + 0.1*dt)/1.1, quindi il primo passo vale ~0.09*dt_CFL e il
     * limite CFL si raggiunge dopo ~30 passi (come il riferimento). */
    {
        static double prev_ref = 0.;
        if (t == 0.) prev_ref = 0.;
        if (dt > prev_ref) dt = (prev_ref + 0.1*dt)/1.1;
        prev_ref = dt;
    }'''

PATCHES = [
    ("P1-viscosita-tensore", OLD_VISC, NEW_VISC,
     "operatore viscoso div(2*mu*D) come viscosity.h (aggiunge grad(div u))"),
]
PATCHES_DT = [
    ("P2-politica-dt", OLD_DT, NEW_DT,
     "limitatore di crescita del dt come basilisk/src/timestep.h"),
]


def read_text(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path, text):
    tmp = path + ".tmp-patch"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def check(src, patches):
    ok = True
    for name, old, new, desc in patches:
        n_old = src.count(old)
        n_new = src.count(new)
        if n_new == 1 and n_old == 0:
            print("  [%s] GIA' APPLICATA (salto) -- %s" % (name, desc))
        elif n_old == 1:
            print("  [%s] OK: 1 occorrenza da sostituire -- %s" % (name, desc))
        else:
            ok = False
            print("  [%s] ERRORE: occorrenze trovate = %d (attese 1) -- %s"
                  % (name, n_old, desc))
            if n_old == 0:
                print("        il file non contiene la stringa attesa: "
                      "e' gia' patchato in modo diverso o la versione non e' "
                      "quella prevista.")
    return ok


def apply_patches(src, patches):
    out = src
    for name, old, new, desc in patches:
        if out.count(new) == 1 and out.count(old) == 0:
            continue
        out = out.replace(old, new, 1)
    return out


def show_context(src, needle, before=2, after=2, label=""):
    idx = src.find(needle)
    if idx < 0:
        print("   (stringa non trovata: %s)" % label)
        return
    start = src.rfind("\n", 0, max(0, idx - 1))
    lines = src[:idx].count("\n")
    seg = src[idx:idx + len(needle)]
    print("   %s a riga %d, %d righe" % (label or "trovato", lines + 1,
                                         seg.count("\n") + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="sorgente da patchare")
    ap.add_argument("--check", action="store_true", help="solo verifica (default)")
    ap.add_argument("--apply", action="store_true", help="applica sul file (con backup)")
    ap.add_argument("--out", default=None,
                    help="scrive la versione patchata in un altro file")
    ap.add_argument("--with-dt-policy", action="store_true",
                    help="include anche P2 (politica del dt del riferimento)")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()

    path = a.file
    if path is None:
        path = DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else FALLBACK_FILE
    if not os.path.exists(path):
        print("ERRORE: file non trovato: %s" % path)
        return 2

    patches = list(PATCHES)
    if a.with_dt_policy:
        patches += PATCHES_DT

    print("file: %s" % path)
    print("patch selezionate: %s" % ", ".join(p[0] for p in patches))
    src = read_text(path)
    print("dimensione: %d byte, %d righe" % (len(src), src.count("\n") + 1))

    print("\n-- verifica stringhe esatte --")
    ok = check(src, patches)
    if not ok:
        print("\nESITO: NON applicabile (nessuna scrittura fatta).")
        return 2

    patched = apply_patches(src, patches)
    if patched == src:
        print("\nESITO: niente da fare (file gia' patchato).")
        return 0

    # controllo di sanita' minimale: bilanciamento delle graffe e DH invariato
    print("\n-- controlli di sanita' --")
    print("   graffe: prima %d/%d, dopo %d/%d"
          % (src.count("{"), src.count("}"), patched.count("{"), patched.count("}")))
    # il file originale puo' avere parentesi non bilanciate nei commenti:
    # si confronta lo SBILANCIAMENTO, non il conteggio assoluto.
    def bal(s):
        return (s.count("(") - s.count(")"), s.count("{") - s.count("}"))
    if bal(src) != bal(patched):
        print("   sbilanciamento prima %s, dopo %s" % (bal(src), bal(patched)))
        print("ESITO: parentesi/graffe sbilanciate -> NON scrivo.")
        return 2
    print("   sbilanciamento parentesi/graffe invariato: %s" % (bal(src),))
    for tag in ("IDX(", "MF_DEV", "bc_u(", "exchange_u(", "MPI_"):
        if patched.count(tag) != src.count(tag):
            print("   ATTENZIONE: cambia il numero di %r (%d -> %d)"
                  % (tag, src.count(tag), patched.count(tag)))

    if a.out:
        write_text(a.out, patched)
        print("\nscritto: %s" % a.out)

    if a.apply:
        if not a.no_backup:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            bak = path + ".bak-" + stamp
            shutil.copy2(path, bak)
            print("backup: %s" % bak)
        write_text(path, patched)
        print("applicato: %s" % path)
    elif not a.out:
        print("\n(dry-run: nessuna scrittura; usare --apply o --out)")

    print("\nESITO: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
