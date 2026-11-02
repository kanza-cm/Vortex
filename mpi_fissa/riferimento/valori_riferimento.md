# Valori di riferimento per la validazione (letti dai file del workspace)

Tutti i numeri di questo file sono **copiati** da file gia' presenti nel
workspace; il percorso e' indicato per ogni tabella. Nessun numero e' stato
misurato nella sessione che ha scritto il prototipo.

## A. t = 0 (stessa IC, stessa Delta)

| quantita | L6 (`MAXLEVEL=6`, 128x128x512) | L8 (`MAXLEVEL=8`, 256x256x1024) |
|---|---|---|
| enstrophy `Z` | 1.25630230e-01 | 1.25961661e-01 |
| energy `E` | 1.00166653e-03 | 1.00165501e-03 |
| umax | 5.74348159e-02 | 5.74350879e-02 |
| wmax | 1.98497629e+00 | 1.99623787e+00 |
| circulation_slice `G` | 5.11479825e-02 | 5.10205626e-02 |
| separation | 2.578125e-01 | 2.539062e-01 |
| xc | 1.289062e-01 | 1.269531e-01 |
| yc | 0.0 | 0.0 |
| cells | 8388608 | 67108864 |
| dt primo passo (solo per info) | 9.907987e-03 | 4.947802e-03 |

Fonti:
- L6: `gpu_nvidia_3d/risultati/error_study/error_study_fissa_3d_gpu/run6/diagnostics.dat` (riga t=0)
- L8: `gpu_nvidia_3d/risultati/diagnostica/diagnostics.dat` (riga t=0) e
  `.../error_study.../run7/diagnostics.dat` (stessi valori: run7 ha
  `NDIV0=2, MAXLEVEL=7` → stessa Delta di L8).

## B. Serie temporale L8 (produzione)

| t | Z | E | umax | wmax | G | sep |
|---|---|---|---|---|---|---|
| 0 | 1.25961661e-01 | 1.00165501e-03 | 5.74350879e-02 | 1.99623787e+00 | 5.10205626e-02 | 2.539062e-01 |
| 1 | 1.25322402e-01 | 9.48302099e-04 | 6.41421825e-02 | 1.96868730e+00 | 5.09828627e-02 | 2.485922e-01 |
| 2 | 1.24729231e-01 | 9.47045861e-04 | 6.12396039e-02 | 1.90748262e+00 | 5.09567782e-02 | 2.511266e-01 |
| 5 | 1.23489544e-01 | 9.43308463e-04 | 5.43031245e-02 | 1.57161760e+00 | 5.09305969e-02 | 2.461248e-01 |
| 10 | 1.25003546e-01 | 9.37079545e-04 | 5.03172763e-02 | 1.43704414e+00 | 5.09193763e-02 | 2.400994e-01 |
| 20 | 1.32974207e-01 | 9.23971063e-04 | 5.40651157e-02 | 2.17353153e+00 | 5.09320982e-02 | 2.461248e-01 |
| 49 | 1.36307836e-01 | 8.84037931e-04 | 5.39599359e-02 | 2.73452950e+00 | 5.09741940e-02 | 3.070322e-01 |
| 99 | 1.36514828e-01 | 8.14752188e-04 | 4.09827083e-02 | 2.30968046e+00 | 5.08477204e-02 | 3.979010e-01 |
| 199 | 1.02464221e-01 | 6.94844697e-04 | 4.11416739e-02 | 1.46307886e+00 | 5.09160087e-02 | 6.236558e-01 |
| 499 | 9.24894884e-02 | 4.11153713e-04 | 4.05759774e-02 | 2.08769155e+00 | 4.84702997e-02 | 6.243771e-01 |
| 999 | 1.62650645e-02 | 1.61108910e-04 | 2.06097383e-02 | 7.29557633e-01 | 3.46116498e-02 | 2.807341e-01 |
| 1499 | 5.27574774e-03 | 1.20148361e-04 | 1.22950356e-02 | 1.19770020e-01 | 2.77752243e-02 | 8.304801e-02 |
| 1500 | 5.27204107e-03 | 1.20094948e-04 | 1.23074688e-02 | 1.19649693e-01 | 2.77637243e-02 | 8.304801e-02 |

Fonte: `gpu_nvidia_3d/risultati/diagnostica/diagnostics.dat` (1502 righe,
t=0..1500). Estratto in `ref_L8_diagnostics_estratto.dat` (stesso formato).

Interpretazione fisica (utile per il confronto qualitativo):
- t=0..20: la perturbazione varicosa fa oscillare enstrophy e separazione; i
  tubi sono ancora distinti;
- t≈20..99: enstrophy sale al massimo (~0.1365) e la separazione cresce fino a
  ~0.4: fase di fusione/riorganizzazione;
- t≈100..500: separazione massima ~0.62, poi decadimento;
- t>500: decadimento viscoso; Z scende a 5e-3 a t=1500.

## C. Cose DA NON usare come riferimento senza verifica

- `.../run6/omega3d_matrix.bin` e `.../run7/omega3d_matrix.bin`: il **tempo**
  del dump non e' scritto nei log (run6 = 13 passi, run7 = 21 passi con
  `MAXLEVEL` diversi → t≈0.1-0.2, non t=10);
- `gpu_nvidia_3d/risultati/matrici/omega3d_matrix.bin`: non se ne conosce
  griglia/tempo dalla sola lista; va aperto (header int32) prima di usarlo.

## D. Valori derivati utili (calcolati a mano dai dati sopra, non misurati)

- `dt` medio a L8 nella fase t=0..10: ~1.8e-2 → ~55 passi per unita' di tempo;
  a t≈1500: dt=8.3e-2 → 12 passi per unita'. Un run L8 fino a t=10 richiede
  ~500-600 passi.
- la produzione ha fatto ~38000 passi in ~1500 unita' di tempo.
