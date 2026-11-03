# Vortici inclinati (TILT=2) — studi a TMAX=30

Configurazione: due tubi di vorticità con perturbazione varicosa e assi
inclinati a X (TILT=2, apertura ~8 gradi).

## Contenuto
- `studio_errore_fissa/`: convergenza a griglia fissa (saetta3d, GPU),
  MAXLEVEL = 4..8 con **L8 come riferimento**, TMAX = 30, CFL = 0.5.
  Output: `run4..run8/`, `error.dat`, `error.png`.
- `heatmap_adattiva/`: mappa di errore con griglia adattiva (incflo/AMReX GPU),
  MINLEVEL = 4, MAXLEVEL = 4..8 con **L8 come riferimento**, TMAX = 30,
  REF_RATIO = 4, soglie VORTERR = "0.3 0.1 3e-2", max_cells = 15e6.
  Output: run dirs `L*_thr*/`, `heatmap.dat`, `heatmap_l2.dat`,
  `heatmap.png`, `heatmap_l2.png`.

## Comandi
```sh
# entrambi gli studi per questa configurazione
./esegui_studio.sh

# oppure separati
TILT=2 LMIN=4 LMAX=8 T=30 CFL=0.5 OUT_DIR=/home/zorka/simulazioni/Test 1/studio_errore_fissa \
  ../../gpu_nvidia_3d/strumenti/studi/error_study_fissa_3d_gpu.sh
TILT=2 MINLEVEL=4 LMIN=4 LMAX=8 TMAX=30 REF_RATIO=4 CFL=0.5 MAX_CELLS=15e6 \
  THR_LIST="0.3 0.1 3e-2" OUT=/home/zorka/simulazioni/Test 1/heatmap_adattiva \
  bash ../../gpu_amr_3d/strumenti/heatmap_adattiva_3d.sh
```
