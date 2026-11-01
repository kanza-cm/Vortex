> NOTA (riordino 2026-10-26): gli script di studio/test sono in `strumenti/`,
> i risultati in `risultati/`, le credenziali in `credenziali/`, la documentazione
> in `documentazione/`. I percorsi citati sotto sono quelli storici.

# HANDOFF — Viscosità disaccoppiata in centered.h (GPU 3D)

Data: 2026-10-16.  Stato: **RISOLTO e validato** (sessione di fix completata;
simulazione di produzione non ancora riavviata).

## Esito (sessione 2026-10-16, fix)

La viscosità disaccoppiata (3 solve Helmholtz scalari) in
`~/basilisk/src/navier-stokes/centered.h`, event `viscous_term`, è stata
corretta ed è ora **attiva di default** (`DECOUPLED_VISCOSITY=1`).
Validazione (GPU fp32, backend `cuda/multigrid3D`, vs i riferimenti originali
in `ref_validazione/`):

| caso | coupled vs rif | **decoupled vs rif** | var |
|---|---|---|---|
| L6, T=0.5 | 1.244e-3 rms | **1.240e-3 rms** | 18 (vs 22) |
| L7, NDIV0=2, T=0.2 | 1.298e-3 rms | **1.294e-3 rms** | 18 (vs 22) |

(criterio rms ≤ ~1.5e-3 ✔). Tempi a cache JIT calda: equivalenti o migliori
(L7/22 passi: decoupled 41.8s CPU vs coupled 45.4s). VRAM: 18 var vs 22 →
~1 GB in meno a MAXLEVEL=8. Backup del file corretto:
`~/centered.h.decoupled-fixed.bak` (versione pre-fix: `~/centered.h.pre-Tfix.bak`).

## Le due cause del bug (per memoria)

Il bisect temporale (dump WZ ogni 0.05) mostrava un errore decoupled-vs-coupled
che cresceva ~linearmente nel tempo (~8e-3/unità) ⇒ errore per passo
sistematico. Due difetti DISTINTI e indipendenti:

1. **Termine `T = mu*grad(div u)` mancante.** L'operatore accoppiato
   `div(mu(grad u + grad u^T)) = mu*lap(u) + mu*grad(div u)`. Il termine T va
   aggiunto esplicitamente al RHS. NON è trascurabile qui perché la divergenza
   CENTRATA del campo u (celle) è O(0.1) (il campo discretamente solenoidale è
   `uf`, non `u`): senza T, errore per passo ~ `mu*grad(div u*)` (~2.5e-6 rms
   in u su dt di produzione). T usa la discretizzazione di
   `viscosity.h/residual_viscosity` (ramo non-TREE). Test isolato su campo di
   produzione L6/dt=0.03: rms coupled-vs-decoupled 6.4e-6 → 2.4e-8 (GPU fp32).
   Restava però un errore LOCALIZZATO di 9.6e-5 max sulle pareti y/z ⇒ causa 2.

2. **Bug del backend GPU con le scritture di UNA SOLA componente.** I kernel
   di `poisson()` aggiornano only `u.x` (o `u.y`, `u.z`): sul backend CUDA
   questo applica ai ghost la BC sbagliata sulle pareti perpendicolari
   (es. ghost di `u.x` alla parete y: `u.x[0,1,0] = -u.x[]` invece di
   `+u.x[]`, cioè no-slip al posto di free-slip). Il solve viene quindi
   eseguito con la BC sbagliata a ridosso delle pareti (errore ~1e-4) e i
   ghost rovinati si propagano. Verifica empirica minima:

   ```c
   foreach() u.x[] = 5.;     // scrittura di una SOLA componente
   // CPU: ghost y di u.x = +5 (free-slip, corretto)
   // GPU: ghost y di u.x = -5 (SBAGLIATO); scrivendo tutte e 3 le
   //      componenti in un solo loop il GPU torna corretto (+5)
   ```

   `poisson()` non è modificabile senza toccare il core; fix adottato:
   **solve su scalari temporanei `wx/wy/wz`** con BC scalari esplicite
   (normale → `dirichlet(0)`, tangenziali → `neumann(0)`) e **copy-back
   full-vector** (`foreach() { u.x[]=wx[]; u.y[]=wy[]; u.z[]=wz[]; }`).
   Con scalari puri nessuna rotazione normale/tangenziale entra in gioco e il
   copy-back scrive le 3 componenti insieme (ghost corretti). Insieme al
   termine T: coupled-vs-decoupled rms 2.4e-8, max 2.6e-7 (identico al
   double CPU); senza = 6.4e-6 rms / 9.6e-5 max.

Nota: il solve scalare converge in 1 ciclo (`mgu.i=1`, resa ~3e-9) come
l'accoppiato, quindi la correttezza delle BC era l'unico problema di
convergenza.

## Dove sta tutto (aggiornato)

| cosa | percorso |
|---|---|
| Solver (fisica, non modificato) | `/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/saetta3d.c` |
| Header riscritto (attivo, fix incluso) | `/home/zorka/basilisk/src/navier-stokes/centered.h` |
| Backup post-fix (validato) | `~/centered.h.decoupled-fixed.bak` |
| Backup pre-fix-T (per confronto) | `~/centered.h.pre-Tfix.bak` |
| Backup pre-fix (riscrittura originale) | `~/centered.h.lowvram.bak` |
| Riferimenti di validazione | `gpu_nvidia_3d/ref_validazione/` |
| Tool di confronto | `ref_validazione/confronta.py nuovo.bin riferimento.bin` |
| Test di debug (visc_debug) | `~/visc_debug/` (sorgenti test isolati operatori/BC e script di bisect e validazione; i dump .bin sono stati rimossi) |

## Come ririprodurre la validazione (comandi)

```sh
# in ~/visc_debug (o ~/tmp): copia saetta3d.c e compila coupled/decoupled
cd ~/visc_debug
export BASILISK=$HOME/basilisk/src; export PATH=$PATH:$BASILISK
CUDA=/usr/local/cuda-13.1/targets/x86_64-linux
export LD_LIBRARY_PATH=$CUDA/lib:$LD_LIBRARY_PATH
qcc -autolink -disable-dimensions -O3 -grid=cuda/multigrid3D \
    -I"$CUDA/include" -L"$CUDA/lib" -L"$CUDA/lib/stubs" \
    -DMOVIE=0 -DVTK=0 -DMATRIX=1 -DMAXLEVEL=6 -DTMAX=0.5 \
    -DDECOUPLED_VISCOSITY=1 saetta3d.c -o t6_dec -lm -lz
mkdir -p run6 && (cd run6 && ../t6_dec)
python3 "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/ref_validazione/confronta.py" \
    run6/omega3d_matrix.bin \
    "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/ref_validazione/rif_orig_L6_T05/omega3d_matrix.bin"
# atteso: rms ~1.24e-3 (criterio ≤1.5e-3)
# per L7: -DNDIV0=2 -DMAXLEVEL=7 -DTMAX=0.2, rif_orig_N2_L7_T02
# (default senza -D: DECOUPLED_VISCOSITY=1, cioè la versione corretta)
```

## Nota JIT (resta valida)

Nuovi argomenti di funzione con tipo `real` chiamati nei kernel possono
rompere il backend (vedi `static void correction(double dt)`). Verificare
sempre il log per `GLSL: error` / `foreach() done on CPU` = fallback CPU
(rallenta tutto). Gli ultimi run non hanno fallback.

## Stato attuale / prossimi passi

- `centered.h` attivo: `DECOUPLED_VISCOSITY=1` (corretto e validato),
  `REUSE_PF=1`, advection con 3 temporanee, prediction con 1 `du`,
  filmati via `p`, `NDIV0`, `MATRIX`, error study.
- `saetta3d.c`: `VTK=0` di default (niente .vti; i filmati li fa MOVIE via
  `output_ppm`; per riattivare: `-DVTK=1`). Binario di produzione
  ricostruito e smoke-testato (L8, no fallback CPU, 68 passi fino a t=1).
- La simulazione di produzione NON è stata riavviata (era stata fermata su
  richiesta). Per ripartire: `cd gpu_nvidia_3d && ./build.sh && ./_avvia_x.sh`
  (attenzione: `_avvia_x.sh` cancella gli output precedenti nel dir) e
  verificare in `log-saetta` che non ci siano fallback CPU.
