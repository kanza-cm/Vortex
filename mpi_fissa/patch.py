#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch.py -- corregge la diagnostica `separation` di mpi_fissa.cu.

BUG: `separation` (colonna 9 di diagnostics.dat) vale 0 quando il massimo di
wz sulla fetta nodo z=2-DH/2 e' raggiunto da piu' celle in modo simmetrico
rispetto al centro del piano: il centroide (px,py) vale (0,0), il test
`x*px + y*py < 0.` diventa `0. < 0.` (falso su ogni cella), m2 resta al suo
inizializzatore 0.0, w2 resta 0 e la formula finale forza sep=0.

La patch (identica per CPU e GPU, np=1 e np=2):
  1. sceglie UNA cella di argmax con regola deterministica e globale
     (x massima, poi y massima) invece del centroide dell'intero insieme;
  2. media solo il plateau LOCALE 3x3 attorno a quella cella (la parte di
     media intesa dal riferimento: rifinitura sub-cella di un plateau
     connesso), quindi px,py non possono piu' collassare sull'origine;
  3. inizializza m2 a -1e300 invece che a 0.0 (un massimo deve essere
     raggiunto: con 0 il caso "tutti negativi" dava di nuovo w2=0 -> sep=0).

Uso (dalla radice del progetto, es. ~/simulazioni/Test\\ 1/mpi_fissa):

    python3 patch.py                     # applica (cerca src/mpi_fissa.cu)
    python3 patch.py percorso/mpi_fissa.cu
    python3 patch.py --check             # verifica soltanto (0=patch presente)
    python3 patch.py --dry-run           # mostra il diff senza scrivere
    python3 patch.py --revert            # ripristina il backup .bak_separation

Codici di uscita: 0 = ok / gia' applicata, 1 = --check negativo, 2 = errore
(ancore non trovate, file illeggibile, contenuto inatteso): in caso di errore
il file NON viene toccato.

Prima di scrivere, il file originale viene salvato in
`<sorgente>.bak_separation` (solo la prima volta: un secondo run non
sovrascrive il backup).
"""

import argparse
import difflib
import io
import os
import shutil
import sys

# --- ancore (stringhe ESATTE presenti nel sorgente non patchato) -------------
ANCHOR_START = "    /* secondo nucleo: posizione del massimo di wz sulla fetta nodo */"
ANCHOR_END = "    d.px = px; d.py = py;"

# controlli di contenuto sullo span che viene sostituito: se una di queste
# stringhe manca, il file non e' quello atteso -> non si tocca nulla.
SANITY = [
    "wz == m1",
    "allreduce_sum(red, 3);",
    "double m2 = 0.;",
    "allreduce_max(&m2, 1);",
    "d.sep = (w1 > 0. && w2 > 0.)",
]
SANITY_MAX_LEN = 4000          # lo span atteso e' ~48 righe: oltre, e' altro

# marcatori che identificano la patch gia' applicata
MARKER_A = "double key1 = -1.;"
MARKER_B = "double key2 = -1.;"

NEW_BLOCK = """    /* ------------------------------------------------------------------
     * Primo nucleo: posizione del massimo di wz sulla fetta nodo.
     *
     * Il massimo puo' essere RAGGIUNTO DA PIU' CELLE: per questo caso (due
     * tubi co-rotanti, IC simmetrica) wz sulla fetta e' pari in x e in y,
     * quindi l'insieme di argmax e' simmetrico rispetto al centro del piano
     * e la sua media vale (0,0).  Con px=py=0 il test x*px+y*py<0 diventa
     * 0<0 (falso su ogni cella), m2 resta al suo inizializzatore, w2 resta 0
     * e la formula finale forza sep=0.
     * Rimedio:
     *  1) si sceglie UNA cella di argmax con regola deterministica e globale
     *     (x massima, poi y massima): il risultato non dipende da np ne' da
     *     quale lato l'arrotondamento rompe la simmetria;
     *  2) la media si fa solo sul plateau LOCALE 3x3 attorno a quella cella
     *     (e' la parte di media intesa dal riferimento: rifinitura sub-cella
     *     di un plateau connesso).
     * ------------------------------------------------------------------ */
    double key1 = -1.;
    if (K0 <= k_node_g && k_node_g < K1) {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                if (WZ_SLICE[(size_t)j*nx + i] != m1) continue;
                double key = (double)i*(double)(nx + 1) + (double)j;  /* esatta in double */
                if (key > key1) key1 = key;
            }
    }
    allreduce_max(&key1, 1);            /* la stessa cella su tutti i rank */
    int i1 = -1, j1 = -1;
    if (key1 >= 0.) {
        i1 = (int)(key1/(double)(nx + 1));
        j1 = (int)(key1 - (double)i1*(double)(nx + 1) + 0.5);
    }
    double px = 0., py = 0., w1 = 0.;
    if (K0 <= k_node_g && k_node_g < K1 && i1 >= 0) {
        for (int j = j1 - 1; j <= j1 + 1; j++)
            for (int i = i1 - 1; i <= i1 + 1; i++) {
                if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
                if (WZ_SLICE[(size_t)j*nx + i] != m1) continue;
                w1 += 1.; px += -0.5 + (i + 0.5)*(double)DH; py += -0.5 + (j + 0.5)*(double)DH;
            }
    }
    double red[3] = {px, py, w1};
    allreduce_sum(red, 3);
    px = red[0]; py = red[1]; w1 = red[2];
    if (w1 > 0.) { px /= w1; py /= w1; }
    /* se il plateau locale e' ancora simmetrico (picco vicino al centro del
     * piano) il centroide cade sull'origine: in quel caso si usa la cella di
     * massimo scelta, che e' un punto ben definito. */
    if (w1 > 0. && px*px + py*py < 0.25*(double)DH*(double)DH) {
        px = -0.5 + (i1 + 0.5)*(double)DH; py = -0.5 + (j1 + 0.5)*(double)DH;
    }

    /* ------------------------------------------------------------------
     * Secondo nucleo: massimo nella meta' opposta del piano.
     * m2 parte da -1e300 e NON da 0: un massimo deve essere raggiunto; con 0
     * il caso "tutti i wz negativi nella meta' opposta" non trovava nessuna
     * cella con wz == m2 => w2=0 => sep=0 (seconda via al valore 0).
     * ------------------------------------------------------------------ */
    double m2 = -1e300;
    if (K0 <= k_node_g && k_node_g < K1) {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                double x = -0.5 + (i + 0.5)*(double)DH, y = -0.5 + (j + 0.5)*(double)DH;
                if (x*px + y*py < 0.) {
                    double wz = WZ_SLICE[(size_t)j*nx + i];
                    if (wz > m2) m2 = wz;
                }
            }
    }
    allreduce_max(&m2, 1);

    double key2 = -1.;
    if (K0 <= k_node_g && k_node_g < K1) {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                double x = -0.5 + (i + 0.5)*(double)DH, y = -0.5 + (j + 0.5)*(double)DH;
                if (x*px + y*py >= 0.) continue;
                if (WZ_SLICE[(size_t)j*nx + i] != m2) continue;
                double key = (double)i*(double)(nx + 1) + (double)j;  /* esatta in double */
                if (key > key2) key2 = key;
            }
    }
    allreduce_max(&key2, 1);
    int i2 = -1, j2 = -1;
    if (key2 >= 0.) {
        i2 = (int)(key2/(double)(nx + 1));
        j2 = (int)(key2 - (double)i2*(double)(nx + 1) + 0.5);
    }
    double qx = 0., qy = 0., w2 = 0.;
    if (K0 <= k_node_g && k_node_g < K1 && i2 >= 0) {
        for (int j = j2 - 1; j <= j2 + 1; j++)
            for (int i = i2 - 1; i <= i2 + 1; i++) {
                if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
                double x = -0.5 + (i + 0.5)*(double)DH, y = -0.5 + (j + 0.5)*(double)DH;
                if (x*px + y*py >= 0.) continue;
                if (WZ_SLICE[(size_t)j*nx + i] != m2) continue;
                w2 += 1.; qx += x; qy += y;
            }
    }
    double red2[3] = {qx, qy, w2};
    allreduce_sum(red2, 3);
    qx = red2[0]; qy = red2[1]; w2 = red2[2];
    if (w2 > 0.) { qx /= w2; qy /= w2; }
    d.sep = (w1 > 0. && w2 > 0.) ? sqrt((px - qx)*(px - qx) + (py - qy)*(py - qy)) : 0.;
    d.px = px; d.py = py;"""


def find_source(explicit):
    """Trova mpi_fissa.cu: argomento esplicito, poi i percorsi noti."""
    cands = []
    if explicit:
        cands.append(explicit)
    cands += [
        os.path.join("src", "mpi_fissa.cu"),
        os.path.join("mpi_fissa", "src", "mpi_fissa.cu"),
        "/home/zorka/simulazioni/Test 1/mpi_fissa/src/mpi_fissa.cu",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c), cands
    return None, cands


def read_src(path):
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_src(path, text):
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, "." + os.path.basename(path) + ".patch.tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def balance(s):
    return s.count("{") - s.count("}")


def is_patched(src):
    return (MARKER_A in src) and (MARKER_B in src)


def locate_span(src):
    """Restituisce (inizio, fine, span) oppure solleva ValueError con motivo."""
    n_start = src.count(ANCHOR_START)
    if n_start != 1:
        raise ValueError("ancora iniziale trovata %d volte (attesa 1):\n  %s"
                         % (n_start, ANCHOR_START))
    i0 = src.index(ANCHOR_START)
    i1 = src.find(ANCHOR_END, i0)
    if i1 < 0:
        raise ValueError("ancora finale non trovata dopo quella iniziale:\n  %s"
                         % ANCHOR_END)
    i1 += len(ANCHOR_END)
    span = src[i0:i1]
    if len(span) > SANITY_MAX_LEN:
        raise ValueError("lo span da sostituire e' lungo %d caratteri (max %d): "
                         "non sembra il blocco atteso" % (len(span), SANITY_MAX_LEN))
    missing = [s for s in SANITY if s not in span]
    if missing:
        raise ValueError("nello span mancano le stringhe attese: %s" % missing)
    if balance(span) != 0:
        raise ValueError("lo span ha graffe sbilanciate (%d): non e' il blocco atteso"
                         % balance(span))
    return i0, i1, span


def print_diff(span, new_block):
    a = span.splitlines(True)
    b = new_block.splitlines(True)
    d = list(difflib.unified_diff(a, b, fromfile="mpi_fissa.cu (prima)",
                                 tofile="mpi_fissa.cu (dopo)", n=2))
    sys.stdout.write("".join(d) + "\n")


def recompute(src, span_start, span_end, new_block):
    return src[:span_start] + new_block + src[span_end:]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Applica la correzione della diagnostica `separation` "
                    "a mpi_fissa.cu (stringhe esatte, tutto-o-niente).")
    ap.add_argument("file", nargs="?", default=None,
                    help="percorso di mpi_fissa.cu (default: src/mpi_fissa.cu)")
    ap.add_argument("--file", dest="file_opt", default=None,
                    help="come l'argomento posizionale")
    ap.add_argument("--check", action="store_true",
                    help="verifica soltanto se la patch e' gia' applicata")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra il diff senza scrivere")
    ap.add_argument("--revert", action="store_true",
                    help="ripristina il sorgente dal backup .bak_separation")
    ap.add_argument("--quiet", action="store_true", help="meno output")
    args = ap.parse_args(argv)

    explicit = args.file_opt or args.file
    path, cands = find_source(explicit)
    if path is None:
        print("ERRORE: mpi_fissa.cu non trovato. Provati:\n  " + "\n  ".join(cands),
              file=sys.stderr)
        return 2
    if not args.quiet:
        print("# sorgente: %s" % path)

    src = read_src(path)
    bak = path + ".bak_separation"

    # ---- revert ---------------------------------------------------------
    if args.revert:
        if not os.path.isfile(bak):
            print("ERRORE: backup %s inesistente" % bak, file=sys.stderr)
            return 2
        if not args.quiet:
            print("# ripristino da %s" % bak)
        write_src(path, read_src(bak))
        print("OK: %s ripristinato dal backup" % path)
        return 0

    # ---- check ----------------------------------------------------------
    if args.check:
        if is_patched(src):
            print("OK: la patch risulta GIA' applicata (marker presenti)")
            return 0
        print("NO: patch NON applicata (mancano %s / %s)" % (MARKER_A, MARKER_B),
              file=sys.stderr)
        return 1

    # ---- applica --------------------------------------------------------
    if is_patched(src):
        print("OK: la patch risulta gia' applicata, nessuna modifica (idempotente)")
        return 0

    try:
        i0, i1, span = locate_span(src)
    except ValueError as e:
        print("ERRORE: %s" % e, file=sys.stderr)
        print("Il file NON e' stato modificato.", file=sys.stderr)
        return 2

    # il blocco nuovo usa \n: se il file e' CRLF lo si converte
    block = NEW_BLOCK.replace("\n", "\r\n") if ("\r\n" in src) else NEW_BLOCK
    new_src = recompute(src, i0, i1, block)
    if balance(block) != 0:
        print("ERRORE: il blocco nuovo ha graffe sbilanciate (%d)"
              % balance(block), file=sys.stderr)
        return 2
    if balance(new_src) != balance(src):
        print("ERRORE: il risultato avrebbe graffe sbilanciate rispetto "
              "all'originale (%d vs %d)" % (balance(new_src), balance(src)),
              file=sys.stderr)
        return 2

    if not args.quiet:
        print_diff(span, block)

    if args.dry_run:
        print("DRY-RUN: nessun file scritto.")
        return 0

    if not os.path.isfile(bak):
        shutil.copyfile(path, bak)
        print("# backup: %s" % bak)
    else:
        print("# backup gia' presente (non sovrascritto): %s" % bak)

    write_src(path, new_src)
    print("OK: patch applicata a %s" % path)
    print("    righe sostituite: %d -> %d"
          % (len(span.splitlines()), len(NEW_BLOCK.splitlines())))
    print()
    print("Prossimi passi suggeriti:")
    print("  NX=128 ./scripts/build.sh cpu && NX=128 ./scripts/build.sh gpu")
    print("  ./scripts/run_cpu.sh 1 -tmax 0.05 -out out/fix_cpu_np1")
    print("  head -3 out/fix_cpu_np1/diagnostics.dat   # sep a t=0 != 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
