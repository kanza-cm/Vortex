# Animazione T=300 — vortici NON inclinati, griglia fissa L8

- solver: `saetta3d.c` (Basilisk centered.h, CUDA, singola precisione)
- configurazione: **TILT=0** (assi paralleli), `NDIV0=1`, **MAXLEVEL=8**
  -> griglia 256 x 256 x 1024 = 67.1 M celle (~7.5 GB VRAM)
- TMAX = 300, CFL = 0.5
- movie: `omega3d_node.mp4` (fetta a z=2), `omega3d_anti.mp4` (fetta a z=0),
  `omega3d.ppm` (frame finale)
- dump finale per ripartire: `chk_300.000.bin` (velocità di faccia + pressione,
  ~1.1 GB); `omega3d_matrix.bin` (omega_z finale per analisi)
- log: `log.txt`

## Comandi
```sh
./esegui.sh              # build + run T=300 (lungo: ~7-9 h su RTX 3070 8GB)
./riparti.sh 300 600     # riparte dal dump a t=300 e continua fino a t=600
```
Nota: il restart ricostruisce il campo a partire da un checkpoint scritto in
singola precisione; nel test L5 la differenza rispetto a una run continua è
~3e-4 rms (0.03% del massimo), sufficiente per continuare l'animazione ma non
bit-exact.
